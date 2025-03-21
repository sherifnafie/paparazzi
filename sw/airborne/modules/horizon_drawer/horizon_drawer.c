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
//#include <opencv2/opencv.hpp> // for more advanced CV

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
static float maxDistance = 2.25f;
static float heading_increment = 5.f;
static const int16_t max_trajectory_confidence = 5;

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

/* We'll also keep track of the "best column" that we found, so we can
 * do something with it in horizon_drawer_periodic (e.g. turn heading that way).
 */
static int best_column = -1; // -1 => none found

/* ------------------------------------------------------------------ */
/*  Canny-like steps                                                  */
/* ------------------------------------------------------------------ */

/**
 * Extract the Y channel from YUV422 into a single grayscale array.
 * Size: w*h, row-major.
 */
static void extractY(const struct image_t *img, uint8_t *gray)
{
  VERBOSE_PRINT("extractY\n");
  uint16_t w = img->w;
  uint16_t h = img->h;
  const uint8_t *buf = img->buf;

  // Each row has w*2 bytes in YUV422
  // We'll simply read each pixel's Y into gray array
  int idx = 0;
  for (int py = 0; py < (int)h; py++) {
    int row_start = py * w * 2;
    for (int px = 0; px < (int)w; px++) {
      if ((px % 2) == 0) {
        // even x => Y at [2*px + 1]
        gray[idx++] = buf[row_start + 2*px + 1];
      } else {
        // odd  x => Y at [2*px - 1]
        gray[idx++] = buf[row_start + 2*px - 1];
      }
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
static void sobel_edge(const uint8_t *gray_in, uint8_t *edge_out, int width, int height)
{
  VERBOSE_PRINT("sobel_edge\n");
  //cv::Canny(gray_in, edge_out, 50, 130); // use this for full Canny


  memset(edge_out, 0, width * height);

  const int EDGE_THRESH = 30; // tune this 80
  for (int y = 1; y < height - 1; y++) {
      for (int x = 1; x < width - 1; x++) {
          int idx = y * width + x;

          int gx = -gray_in[(y-1)*width + (x-1)] + gray_in[(y-1)*width + (x+1)]
                   -2 * gray_in[y*width + (x-1)] + 2 * gray_in[y*width + (x+1)]
                   -gray_in[(y+1)*width + (x-1)] + gray_in[(y+1)*width + (x+1)];

          int gy = gray_in[(y-1)*width + (x-1)] + 2 * gray_in[(y-1)*width + x] + gray_in[(y-1)*width + (x+1)]
                   -gray_in[(y+1)*width + (x-1)] - 2 * gray_in[(y+1)*width + x] - gray_in[(y+1)*width + (x+1)];

          int mag = abs(gx) + abs(gy); // Approximation of magnitude

          if (mag > EDGE_THRESH) {
            edge_out[idx] = 255;
          } else {
            edge_out[idx] = 0;
          }
      } 
  }


  // memset(edge_out, 0, w*h); // initialize to 0
  // // For simplicity, skip the border
  // const int EDGE_THRESH = 30; // tune this 80

  // for (int y = 1; y < h-1; y++) {
  //   for (int x = 1; x < w-1; x++) {
  //     // index in row-major
  //     int idx = y*w + x;

  //     // Sobel Gx, Gy
  //     // sample neighbors
  //     int v00 = gray_in[(y-1)*w + (x-1)];
  //     int v01 = gray_in[(y-1)*w + x];
  //     int v02 = gray_in[(y-1)*w + (x+1)];
  //     int v10 = gray_in[ y   *w + (x-1)];
  //     int v12 = gray_in[ y   *w + (x+1)];
  //     int v20 = gray_in[(y+1)*w + (x-1)];
  //     int v21 = gray_in[(y+1)*w + x];
  //     int v22 = gray_in[(y+1)*w + (x+1)];

  //     int gx =  ( -v00 + v02
  //               -2*v10 + 2*v12
  //               -v20 + v22 );
  //     int gy =  (  v00 + 2*v01 + v02
  //               - v20 - 2*v21 - v22 );

  //     int mag = abs(gx) + abs(gy); // simpler than sqrt(gx^2+gy^2)
  //     //VERBOSE_PRINT("mag: %i \n", mag);
  //     if (mag > EDGE_THRESH) {
  //       edge_out[idx] = 255;
  //     } else {
  //       edge_out[idx] = 0;
  //     }
  //   }
  // }
}

/**
 * For each column, we scan from bottom to top, look for the first edge_out[y*w + x] = 255.
 * The "distance" is (h-1 - y). We pick the column that yields the largest distance.
 *
 * If no edges at all, or if every column has edges right near the bottom, 
 * we might declare "unsafe."
 */
static int find_best_column(const uint8_t *edge, int w, int h, int *best_dist)
{
  
  int best_col = -1;
  int best_val = -1; // the best distance so far

  for (int x = 0; x < w; x++) {
    // start from bottom row = h-1, go upward
    int dist = 0;
    bool found_edge = false;
    for (int y = h/2-1; y >= 0; y--) { //only search in the lower half
      int idx = y*w + x;
      if (edge[idx] == 255) {
        // found an edge => measure distance from bottom
        dist = (h-1) - y; 
        found_edge = true;
        break;
      }
    }
    if (!found_edge) {
      // means no edge in this column => effectively dist = h 
      // or interpret that as "completely free"
      dist = h/2;
    }
    if (dist > best_val) {
      best_val = dist;
      best_col = x;
    }
  }
  *best_dist = best_val;
  VERBOSE_PRINT("find_best_column. Best_col: %i with best_dist: %i\n", best_col, best_val);
  return best_col;
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
    return img;
  }

  uint16_t w = img->w;
  uint16_t h = img->h;

  // 1) Extract Y channel
  static uint8_t gray[2000*2000]; // Be sure this is big enough for your max resolution!
  if (w*h > 2000*2000) {
    // safety check: if your cam is bigger than 2000x2000, you need a bigger buffer
    horizon_black_percent = 0.f;
    best_column = -1;
    return img;
  }
  extractY(img, gray);

  // 2) Sobel Edge
  static uint8_t edges[2000*2000];
  sobel_edge(gray, edges, w, h);

  // 3) Find best column
  int best_dist = 0;
  int col = find_best_column(edges, w, h, &best_dist);

  best_column = col; // store globally
  // If best_dist is very small, or best_col = -1 => "unsafe"
  // If we found a big gap => "safe"

  // Let's say if best_dist > ~1/4 of the height => safe
  int min_safe_dist = h / 4;  
  if (best_dist < min_safe_dist) {
    horizon_black_percent = 0.f;   // "unsafe"
  } else {
    horizon_black_percent = 100.f; // "safe"
  }

  VERBOSE_PRINT("Canny-like best_col=%d best_dist=%d => black%%=%.1f\n",
                best_column, best_dist, horizon_black_percent);

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
  best_column = -1;
}

/* ------------------------------------------------------------------ */
/*                    horizon_drawer_periodic()                       */
/* ------------------------------------------------------------------ */
void horizon_drawer_periodic(void)
{
  if (!autopilot_in_flight()) {
    return;
  }

  float black_percent = horizon_black_percent;

  VERBOSE_PRINT("Edges => black%%=%.1f, threshold=%.1f, state=%d, best_col=%d\n",
                black_percent, horizon_threshold, navigation_state, best_column);

  // If black_percent >= threshold => "safe"
  if (black_percent >= horizon_threshold) {
    obstacle_free_confidence++;
  } else {
    obstacle_free_confidence -= 2;
  }

  // Bound obstacle_free_confidence
  if (obstacle_free_confidence < 0) {
    obstacle_free_confidence = 0;
  } else if (obstacle_free_confidence > max_trajectory_confidence) {
    obstacle_free_confidence = max_trajectory_confidence;
  }

  float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

  switch (navigation_state) {

    case SAFE:
    VERBOSE_PRINT("SAFE: best_col=%d\n", best_column);
      // In principle, we might want to steer toward best_column here if it's good
      // but let's keep your old logic:
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
    VERBOSE_PRINT("Obstacle Found: best_col=%d\n", best_column);
      // Stop => place WP_GOAL, WP_RETREAT, WP_TRAJECTORY at current pos
      waypoint_move_here_2d(WP_GOAL);
      waypoint_move_here_2d(WP_RETREAT);
      waypoint_move_here_2d(WP_TRAJECTORY);

      chooseRandomIncrementAvoidance();
      navigation_state = SEARCH_FOR_SAFE_HEADING;
      break;

    case SEARCH_FOR_SAFE_HEADING:
    VERBOSE_PRINT("Search for Safe Heading: best_col=%d\n", best_column);
      increase_nav_heading(heading_increment);

      // For a more direct approach, you could set heading based on best_column:
      // e.g.  float new_heading = headingFromColumn(best_column);
      //       nav.heading = new_heading;

      if (obstacle_free_confidence >= 2) {
        navigation_state = SAFE;
      }
      break;

    case OUT_OF_BOUNDS:
    VERBOSE_PRINT("Out of Bounds: best_col=%d\n", best_column);
      increase_nav_heading(heading_increment);
      // WP_Trajecktory says to the function that the WP which is next approach meant
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

