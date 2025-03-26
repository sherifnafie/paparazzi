/**
 * @file "modules/horizon_drawer/horizon_drawer.c"
 *
 * Example that uses a simple "Canny-like" edge detection. We scan each column
 * from the bottom up to find the first edge. The column with the largest gap
 * from bottom to edge is considered "safest." If no edges are found, or everything
 * is basically blocked near the bottom, we say "unsafe."
 *
 * The rest of the code is the same state machine from your previous horizon_drawer.
 */

#include <time.h>
#include <stdio.h>
#include <stdlib.h>    // rand(), srand()
#include <math.h>      // sinf, cosf, fminf
#include <stdbool.h>   // for bool
#include <string.h>    // for memset if needed
#include <pthread.h>

/* Paparazzi / firmware includes */
#include "modules/horizon_drawer/horizon_drawer.h"
#include "modules/computer_vision/cv.h"     // cv_add_to_device()
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"             // front_camera references
#include "generated/flight_plan.h"          // WP_GOAL, WP_RETREAT, etc.
#include "state.h"                          // stateGetPositionEnu_i/f()
#include "modules/core/abi.h"               // if you need ABI

/* ------------------------------------------------------------------ */
/*    Macros for debugging prints                                     */
/* ------------------------------------------------------------------ */
#define HORIZON_DRAWER_VERBOSE true

#define PRINT(string, ...) \
  fprintf(stderr, "[horizon_drawer->%s()] " string, __FUNCTION__, ##__VA_ARGS__)

#if HORIZON_DRAWER_VERBOSE
  #define VERBOSE_PRINT PRINT
#else
  #define VERBOSE_PRINT(...)
#endif

/* ------------------------------------------------------------------ */
/*  The old 'horizon_black_percent' now acts as a simple 0/100 flag   */
/*  for the rest of the state machine: >=50 => "safe", else "unsafe." */
/* ------------------------------------------------------------------ */
static float horizon_black_percent = 0.0f;
float horizon_threshold = 50.0f; // if horizon_black_percent >= 50 => safe

/* Confidence tracking and state machine */
static int16_t obstacle_free_confidence = 0;
static int16_t best_direction_global = 0;
static float maxDistance = 2.25f;
static float heading_increment = 5.f;
static float heading_increment_obstacle_found = 5.f;
static const int16_t max_trajectory_confidence = 4;
static bool possible_obstacle_in_center = false;

/* Extra offset (in degrees) to steer left or right */
static float extra_heading_offset = 0.0f;

static const int middle_danger_zone = 150; // The considered width of the middle in pixels, 52 = 10% of 520
static const int min_safe_dist = 45; // Minimally required distance between bottom and first edge in the middle to be SAFE 

static const int middle_start = 520 / 2 - middle_danger_zone / 2;
static const int middle_end = 520 / 2 + middle_danger_zone / 2;

/* Navigation states */
enum navigation_state_t {
  SAFE,
  OBSTACLE_FOUND,
  SEARCH_FOR_SAFE_HEADING,
  OUT_OF_BOUNDS
};
static enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;

/* ------------------------------------------------------------------ */
/*     Forward Declarations of Helpers                                */
/* ------------------------------------------------------------------ */
static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static uint8_t chooseRandomIncrementAvoidance(void);
static float chooseBestDirectionChange(int best_direction);

static int measureSingleColumnDistance(struct image_t *img, const uint8_t *edge, int w, int h, int col, int offset_y);

/* We'll also keep track of the "best column" that we found, so we can
 * do something with it in horizon_drawer_periodic (e.g. turn heading that way).
 */
static int best_column = -1; // -1 => none found

static pthread_mutex_t possible_obstacle_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t best_direction_mutex = PTHREAD_MUTEX_INITIALIZER;


/* ------------------------------------------------------------------ */
/*  Canny-like steps                                                  */
/* ------------------------------------------------------------------ */

/**
 * Extract the Y channel from YUV422 into a single grayscale array.
 * Size: w*h, row-major.
 */
static void extractY(const struct image_t *img, uint8_t *gray, int offset_y, int adjusted_h)
{
  uint16_t w = img->w;
  uint16_t h = img->h;
  const uint8_t *buf = img->buf;

  // Adjust the height to exclude the top and bottom #### IMAGE IS ROTATED SO WE ONLY TAKE MIDDLE ROWS
  // Each row has w*2 bytes in YUV422
  // We'll simply read each pixel's Y into gray array
  int idx = 0;
  for (int py = offset_y; py < (int)h - offset_y; py++) {
    int row_start = py * w * 2;
    for (int px = 0; px < (int)w; px++) {
        gray[idx++] = buf[row_start + 2 * px];
    }
  }
  
}

/**
 * A very simplified "Sobel magnitude" approach:
 *   - We compute dX, dY with 3x3 Sobel kernels
 *   - magnitude = sqrt(dX^2 + dY^2)
 *   - We threshold it => edge=255 if magnitude>EDGE_THRESH else 0
 * This is not a full Canny pipeline (no non-max suppression, no hysteresis).
 */
static void sobel_edge(const uint8_t *gray_in, uint8_t *edge_out, int w, int h, int offset_y, int adjusted_h)
{
  memset(edge_out, 0, w*adjusted_h); // initialize to 0
  // For simplicity, skip the border
  const int EDGE_THRESH = 88; // tune this

  // Adjust the height to exclude the top and bottom #### IMAGE IS ROTATED SO WE ONLY TAKE MIDDLE ROWS
  for (int y = 1; y < adjusted_h - 1; y++) {
    for (int x = 1; x < w - 1; x++) {
      int idx = y * w + x;

      // sample neighbors
      int v00 = gray_in[(y-1)*w + (x-1)];
      int v01 = gray_in[(y-1)*w + x];
      int v02 = gray_in[(y-1)*w + (x+1)];
      int v10 = gray_in[ y   *w + (x-1)];
      int v12 = gray_in[ y   *w + (x+1)];
      int v20 = gray_in[(y+1)*w + (x-1)];
      int v21 = gray_in[(y+1)*w + x];
      int v22 = gray_in[(y+1)*w + (x+1)];

      int gx =  ( -v00 + v02
                -2*v10 + 2*v12
                -v20 + v22 );
      int gy =  (  v00 + 2*v01 + v02
                - v20 - 2*v21 - v22 );

      int mag = abs(gx) + abs(gy); // simpler than sqrt(gx^2+gy^2)

      if (mag > EDGE_THRESH) {
        edge_out[idx] = 255;
      } else {
        edge_out[idx] = 0;
      }
    }
  }
}

/**
 * For each column, we scan from bottom to top, look for the first edge_out[y*w + x] = 255.
 * The "distance" is (h-1 - y).
 * We pick the column that yields the largest distance,
 * ignoring columns that exceed 55% => "unsafe => distance=0"
 */
// static int find_best_column(struct image_t *img, const uint8_t *edge, int w, int h, int *best_dist)
// {
//   int best_row = -1;
//   int best_val = -1;

//   for (int y = 0; y < h; y++) {
//     int dist = measureSingleColumnDistance(img, edge, w, h, y);
//     if (dist > best_val) {
//       best_val = dist;
//       best_row = y;
//     }
//   }

//   // Set the entire best_row to 255 in the input image
//   if (best_row >= 0) {
//     uint8_t *buf = (uint8_t *)img->buf;
//     for (int x = 0; x < w; x++) {
//       buf[best_row * w * 2 + x * 2 + 1] = 255; // Y1
//     }
//   }

//   *best_dist = best_val;
//   return best_row;
// }

static int find_best_column(struct image_t *img, const uint8_t *edge, int w, int h, int *best_dist, int *best_direction, int *worst_direction, int *worst_dist, int offset_y, int adjusted_h)
{
  // Adjust the height to exclude the top and bottom, which are actually the sides since the image is rotated
  // Calculate the height for each column
  int *column_heights = (int *)malloc(adjusted_h * sizeof(int));
  for (int y = 0; y < adjusted_h; y++) {
    column_heights[y] = measureSingleColumnDistance(img, edge, w, h, y, offset_y);
  }
  //Initialize
  int max_avg_height = 0;
  int best_row = 0;
  int num_neighbors = 10; // specify the number of neighboring columns to consider

  // Best Direction
  for (int y = 0; y < adjusted_h - num_neighbors + 1; y++) {
    int sum_height = 0;
    for (int n = 0; n < num_neighbors; n++) {
      sum_height += column_heights[y + n];
    }
    int avg_height = sum_height / num_neighbors;
    if (avg_height > max_avg_height) {
      max_avg_height = avg_height;
      best_row = y + num_neighbors / 2; // center of the neighboring columns
    }
  }
  best_row += offset_y;
  // Set the entire best_row to 255 in the input image
  if (best_row >= 0) {
    uint8_t *buf = (uint8_t *)img->buf;
    for (int x = 0; x < w; x++) {
      buf[best_row * w * 2 + x * 2 + 1] = 149; // Y (brightness for green)
      buf[best_row * w * 2 + x * 2 + 0] = 43;  // U (chrominance for green)
      buf[best_row * w * 2 + x * 2 + 3] = 21;  // V (chrominance for green)
    }
  }

  // Worst Direction
  int min_avg_height = 255;
  int worst_row = 0;
  for (int y = middle_start-offset_y; y <= middle_start-offset_y + middle_danger_zone - num_neighbors + 1; y++) {
    int sum_height = 0;
    int valid_neighbors = 0;

    for (int n = 0; n < num_neighbors; n++) {
      if (column_heights[y + n] > 0) { // Only consider valid heights
        sum_height += column_heights[y + n];
        valid_neighbors++;
      }
    }

    if (valid_neighbors > 0) { // Avoid division by zero
      int avg_height = sum_height / valid_neighbors;

      if (avg_height < min_avg_height) {
        min_avg_height = avg_height;
        worst_row = y + num_neighbors / 2; // center of the neighboring columns
      }
    }
  }
  
  worst_row += offset_y;
  // Set the entire best_row to 255 in the input image
  if (worst_row >= 0) {
    uint8_t *buf = (uint8_t *)img->buf;
    for (int x = 0; x < w; x++) {
      buf[worst_row * w * 2 + x * 2 + 1] = 255; // Y1
    }
  }

  *worst_direction = worst_row;
  *best_dist = max_avg_height;
  *worst_dist = min_avg_height;
  free(column_heights);
  *best_direction = best_row;

}

/* ------------------------------------------------------------------ */
/*  Our CV callback: horizon_drawer_detect()                          */
/* ------------------------------------------------------------------ */
static struct image_t *horizon_drawer_detect(struct image_t *img, uint8_t cam_id)
{
  (void)cam_id;
  if (!img || !img->buf) {
    horizon_black_percent = 50.f; // fallback
    best_column = -1;
    extra_heading_offset = 0.0f;
    return img;
  }

  uint16_t w = img->w;
  uint16_t h = img->h;
  float skip_percentage = 0.1f; // skip the top and bottom 20%
  int adjusted_h = h * (1.0f - 2.0f * skip_percentage);
  int offset_y = h * skip_percentage;

  // 1) Extract Y channel
  static uint8_t gray[2000*2000];
  if (w*h > 2000*2000) {
    horizon_black_percent = 0.f;
    best_column = -1;
    extra_heading_offset = 0.0f;
    return img;
  }
  extractY(img, gray, offset_y, adjusted_h);

  // 2) Sobel Edge
  static uint8_t edges[2000*2000];
  sobel_edge(gray, edges, w, h, offset_y, adjusted_h);

  // // Put pixels with edges to y = 255 in the *img  ### FOR DEBUGGING
  // uint8_t *buf = (uint8_t *)img->buf;
  // for (int y = 0; y < adjusted_h; y++) {
  //     for (int x = 0; x < w; x++) {
  //         int idx = y * w + x;
  //         if (edges[idx] == 255) {
  //             buf[ (y + offset_y) * w * 2 + x * 2 + 1] = 255; // Y1
  //         }
  //     }
  // }
  
  // 3) Find best row ## 90 DEGREES TURNED!!
  int best_dist = 0;              // 0 - cut_off_limit * 240
  int best_direction = 0;         // 0 - 520
  int worst_direction = 0;        // 0 - 520
  int worst_dist = 0;             // 0 - cut_off_limit * 240
  find_best_column(img, edges, w, h, &best_dist, &best_direction, &worst_direction, &worst_dist, offset_y, adjusted_h);

  // VERBOSE_PRINT("Worst dist: %d, Best dist: %d\n", worst_dist, best_dist);

  pthread_mutex_lock(&best_direction_mutex);
  best_direction_global = best_direction;
  pthread_mutex_unlock(&best_direction_mutex);

  pthread_mutex_lock(&possible_obstacle_mutex);
  possible_obstacle_in_center = false;
  if (worst_direction >= middle_start && worst_direction <= middle_end) {
    if (worst_dist < min_safe_dist) {
      possible_obstacle_in_center = true;
    }
  }
  pthread_mutex_unlock(&possible_obstacle_mutex);

  // Draw bounding lines for the middle danger zone
  uint8_t *buf = (uint8_t *)img->buf;

  // Draw the top boundary of the middle danger zone
  for (int x = 0; x < min_safe_dist; x++) {
    buf[(middle_start) * w * 2 + x * 2 + 1] = 29;  // Y (brightness for blue)
    buf[(middle_start) * w * 2 + x * 2 + 0] = 255; // U (chrominance for blue)
    buf[(middle_start) * w * 2 + x * 2 + 3] = 107; // V (chrominance for blue)

    buf[(middle_end) * w * 2 + x * 2 + 1] = 29;  // Y (brightness for blue)
    buf[(middle_end) * w * 2 + x * 2 + 0] = 255; // U (chrominance for blue)
    buf[(middle_end) * w * 2 + x * 2 + 3] = 107; // V (chrominance for blue)
  }
  // Draw the sides of the middle danger zone bounding box
  for (int y = middle_start; y <= middle_end; y++) {
    buf[y * w * 2 + min_safe_dist * 2 + 1] = 29;  // Y (brightness for blue)
    buf[y * w * 2 + min_safe_dist * 2 + 0] = 255; // U (chrominance for blue)
    buf[y * w * 2 + min_safe_dist * 2 + 3] = 107; // V (chrominance for blue)
  }


  // ----------------------------------------------------------------
  //   EXTRA STEERING LOGIC:
  //   We look at the columns to the immediate left/right of best_column
  //   whichever side has a bigger distance => tilt slightly that way
  // ----------------------------------------------------------------
  extra_heading_offset = 0.0f; // default
  if (best_column >= 0) {
    int left_col = best_column - 1;
    int right_col = best_column + 1;

    int dist_left = 0;
    if (left_col >= 0) {
      dist_left = measureSingleColumnDistance(img, edges, w, h, left_col, offset_y);
    }

    int dist_right = 0;
    if (right_col < w) {
      dist_right = measureSingleColumnDistance(img, edges, w, h, right_col, offset_y);
    }

    // If left is bigger => turn left a bit
    // If right is bigger => turn right a bit
    // pick an angle, e.g. 3 deg
    if (dist_left > dist_right) {
      extra_heading_offset = +5.f;  // steer a bit left
    } else if (dist_right > dist_left) {
      extra_heading_offset = -5.f;  // steer a bit right
    }
    // if equal => 0
  }

  return img; // must return the image pointer
}

/* ------------------------------------------------------------------ */
/*                     horizon_drawer_init()                          */
/* ------------------------------------------------------------------ */
void horizon_drawer_init(void)
{
  srand(time(NULL));
  chooseRandomIncrementAvoidance();

  VERBOSE_PRINT("Module initialized.\n");

  // Register this module with the camera pipeline
  cv_add_to_device(&front_camera, horizon_drawer_detect, 0, 0);

  // Start out searching
  navigation_state = SEARCH_FOR_SAFE_HEADING;
  obstacle_free_confidence = 0;
  heading_increment_obstacle_found = 0;
  best_column = -1;
  extra_heading_offset = 0.0f;
  possible_obstacle_in_center = false;
}

/* ------------------------------------------------------------------ */
/*                    horizon_drawer_periodic()                       */
/* ------------------------------------------------------------------ */
void horizon_drawer_periodic(void)
{
  if (!autopilot_in_flight()) {
    return;
  }

  pthread_mutex_lock(&possible_obstacle_mutex);
  if (possible_obstacle_in_center) {
    obstacle_free_confidence -= 2;
    VERBOSE_PRINT("Possible obstacle in center, confidence: %d\n", obstacle_free_confidence);
  } else {
    obstacle_free_confidence++;
  }
  pthread_mutex_unlock(&possible_obstacle_mutex);

  pthread_mutex_lock(&best_direction_mutex);
  int best_direction_local = best_direction_global;
  pthread_mutex_unlock(&best_direction_mutex);

  // Bound obstacle_free_confidence
  if (obstacle_free_confidence < 0) {
    obstacle_free_confidence = 0;
  } else if (obstacle_free_confidence > max_trajectory_confidence) {
    obstacle_free_confidence = max_trajectory_confidence;
  }

  VERBOSE_PRINT("Confidence: %d\n", obstacle_free_confidence);

  float moveDistance = fminf(maxDistance, 0.25f * obstacle_free_confidence);

  switch (navigation_state) {

    case SAFE:
      // Optionally steer extra, based on the best_column neighbors
      // e.g. we do it once per iteration
      // increase_nav_heading(extra_heading_offset);

      // In principle, we might want to steer toward best_column if it's good,
      // but let's keep your old logic for waypoint movement:
      moveWaypointForward(WP_TRAJECTORY, 1.5f * moveDistance);

      if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        navigation_state = OUT_OF_BOUNDS;
      }
      else if (obstacle_free_confidence == 0) {
        navigation_state = OBSTACLE_FOUND;
      }
      else {
        moveWaypointForward(WP_GOAL, moveDistance);
        moveWaypointForward(WP_RETREAT, -1.0f * moveDistance);
      }
      break;

    case OBSTACLE_FOUND:
      // Stop => place WP_GOAL, WP_RETREAT, WP_TRAJECTORY at current pos
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_RETREAT);
      waypoint_move_here_2d(WP_TRAJECTORY);

      chooseRandomIncrementAvoidance();
      heading_increment_obstacle_found = chooseBestDirectionChange(best_direction_local);
      navigation_state = SEARCH_FOR_SAFE_HEADING;
      break;

    case SEARCH_FOR_SAFE_HEADING:
      increase_nav_heading(heading_increment);

      if (obstacle_free_confidence >= 2) {
        navigation_state = SAFE;
      }
      break;

    case OUT_OF_BOUNDS:
      increase_nav_heading(heading_increment);
      moveWaypointForward(WP_TRAJECTORY, 1.5f);
      moveWaypointForward(WP_RETREAT, -1.0f);

      if (InsideObstacleZone(WaypointX(WP_TRAJECTORY), WaypointY(WP_TRAJECTORY))) {
        increase_nav_heading(heading_increment);
        obstacle_free_confidence = 0;
        navigation_state = SEARCH_FOR_SAFE_HEADING;
      }
      break;

    default:
      break;
  }
}

/* ------------------------------------------------------------------ */
/*                       Helper Functions                             */
/* ------------------------------------------------------------------ */

/**
 * measureSingleColumnDistance():
 *   We replicate the same logic from find_best_column for a single column,
 *   including the 55% "unsafe" filter.
 */

static int measureSingleColumnDistance(struct image_t *img, const uint8_t *edge, int w, int h, int row, int offset_y)
{
  const float unsafe_limit = 0.55f * (float)w;
  int dist = 0;
  bool found_edge = false;

  // scan from left to right
  uint8_t *buf = (uint8_t *)img->buf;
  for (int x = 0; x < w; x++) {
    int idx = row * w + x;
    if (edge[idx] == 255) {
      dist = x;
      buf[(row + offset_y) * w * 2 + x * 2 + 1] = 255; // Y1
      found_edge = true;
      break;
    }
  }

  if (!found_edge) {
    dist = w; // no edge => effectively the entire row is free
  }

  // If distance >= 55% => set dist=0 => "unsafe"
  if ((float)dist >= unsafe_limit) {
    dist = 0;
  }

  return dist;
}

static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  calculateForwards(&new_coor, distanceMeters);
  moveWaypoint(waypoint, &new_coor);
  return 0;
}

static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading = stateGetNedToBodyEulers_f()->psi;

  new_coor->x = stateGetPositionEnu_i()->x +
                POS_BFP_OF_REAL(sinf(heading) * distanceMeters);
  new_coor->y = stateGetPositionEnu_i()->y +
                POS_BFP_OF_REAL(cosf(heading) * distanceMeters);

  VERBOSE_PRINT("calcForward: +%.2fm => (%.2f,%.2f)\n",
                distanceMeters,
                POS_FLOAT_OF_BFP(new_coor->x),
                POS_FLOAT_OF_BFP(new_coor->y));
  return 0;
}

static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  VERBOSE_PRINT("moveWaypoint #%d => (%.2f,%.2f)\n",
                waypoint,
                POS_FLOAT_OF_BFP(new_coor->x),
                POS_FLOAT_OF_BFP(new_coor->y));

  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return 0;
}

static uint8_t increase_nav_heading(float incrementDegrees)
{
  float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
  FLOAT_ANGLE_NORMALIZE(new_heading);
  nav.heading = new_heading;

  VERBOSE_PRINT("increase_nav_heading: heading => %.2f deg\n", DegOfRad(new_heading));
  return 0;
}

static uint8_t chooseRandomIncrementAvoidance(void)
{
  if (rand() % 2 == 0) {
    heading_increment = 5.f;
  } else {
    heading_increment = -5.f;
  }
  VERBOSE_PRINT("chooseRandomIncrement: heading_increment=%.2f\n", heading_increment);
  return 0;
}

static float chooseBestDirectionChange(int best_direction)
{
  float heading_increment_obstacle_found = 0.0f;
  if (best_direction >= 50) {
    heading_increment_obstacle_found = (((float)best_direction - 50.0f) / 50.0f) * 45.0f;
  } else {
    heading_increment_obstacle_found = ((float)best_direction / 50.0f) * 45.0f;
  }
  VERBOSE_PRINT("chooseRandomIncrement: heading_increment=%.2f\n", heading_increment_obstacle_found);
  return heading_increment_obstacle_found;
}

