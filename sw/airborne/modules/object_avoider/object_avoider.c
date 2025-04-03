// This file is the main file for the object avoider module. It is responsible for avoiding obstacles in the drone's path.
// The module uses a neural network (outputting a safety grid) to perceive the environment
// and a Vector Field Histogram (VFH) inspired approach to calculate steering commands.

#include "modules/object_avoider/object_avoider.h"
#include "firmwares/rotorcraft/navigation.h"
#include "generated/airframe.h"
#include "state.h"
#include "modules/core/abi.h"
#include <time.h>
#include <stdio.h>
#include <math.h> // Include for trigonometric functions, fabs, sqrt, M_PI
#include <float.h> // Include for FLT_EPSILON

#define NAV_C // needed to get the nav functions like Inside...
#include "generated/flight_plan.h" // Includes InsideObstacleZone, WaypointX/Y, waypoint_* functions
#include "math/pprz_algebra_float.h" // For FLOAT_ANGLE_NORMALIZE if needed (check if included elsewhere)


#define ORANGE_AVOIDER_VERBOSE TRUE

#define PRINT(string,...) fprintf(stderr, "[object_avoider->%s()] " string,__FUNCTION__ , ##__VA_ARGS__)
#if ORANGE_AVOIDER_VERBOSE
#define VERBOSE_PRINT PRINT
#else
#define VERBOSE_PRINT(...)
#endif


#ifndef TENSOR_OUTPUT_id
#define TENSOR_OUTPUT_id ABI_BROADCAST
#endif

// REPULSION_EXPONENT: Controls how sensitive steering is to obstacles. Lower (e.g., 1.0) = reacts mainly to close things. Higher (e.g., 2.0) = reacts strongly even to things with moderately low ratings (further away or less dense).

// GOAL_STRENGTH: Controls the "desire" to go forward. Higher = needs stronger/more obstacles to divert. Lower = more easily diverted. Tune relative to REPULSION_EXPONENT.

// MIN/MAX_FORWARD_SPEED: Direct control over speed limits based on forward clearance.

// CENTRAL_COL_START/END: Defines the "straight ahead" region for speed control.

// OUT_OF_BOUNDS_TURN_DEG, OUT_OF_BOUNDS_PROBE_DIST: Control the fixed recovery maneuver when outside the zone. PROBE_DIST needs to be large enough to actually cross back inside.

// WAYPOINT_DISTANCE: Base distance for the local target when moving slowly. Affects responsiveness.

// MAX_HEADING_CHANGE_DEG: Limits how sharply it can turn in one cycle, affecting smoothness.

// --- Configuration Constants ---
// Grid dimensions (must match CNN output)
#define GRID_COLUMNS 32
#define GRID_ROWS 12 // Full grid rows
#define ROWS_TO_CONSIDER 9 // Number of top rows to average for column rating

// VFH Parameters (TUNING NEEDED HERE)
#define FOV_DEGREES 180.0f          // Horizontal Field of View covered by the 32 columns
#define MAX_RATING 255.0f           // Max possible safety rating value from CNN grid interpretation
// *** TUNING: How strongly obstacles repel. Lower values react less to distant objects. Try 1.0 to 1.5 ***
#define REPULSION_EXPONENT 1.6f
// *** TUNING: Attraction force towards moving forward. Higher values make it push forward more aggressively. Try 50.0 to 150.0 ***
#define GOAL_STRENGTH 150.0f
#define MAX_HEADING_CHANGE_DEG 45.0f // Max steering angle change per update cycle [deg]

// Speed Control Parameters (TUNING NEEDED HERE)
// *** TUNING: Adjust max/min speed based on drone capability and environment ***
#define MAX_FORWARD_SPEED 1.75f      // Max speed in clear space [m/s]
#define MIN_FORWARD_SPEED 0.2f      // Min speed when navigating obstacles [m/s]
#define CENTRAL_COL_START 13        // Start column index for forward clearance check
#define CENTRAL_COL_END 18          // End column index for forward clearance check

// Safety & Navigation Parameters (TUNING NEEDED HERE)
#define WAYPOINT_DISTANCE 0.75f     // Base distance ahead to place the local navigation waypoint [m]
// *** TUNING: Angle to turn when hitting boundary ***
#define OUT_OF_BOUNDS_TURN_DEG 45.0f
// *** TUNING: Distance to move forward after turning when OOB. Should be large enough to re-enter zone. ***
#define OUT_OF_BOUNDS_PROBE_DIST (WAYPOINT_DISTANCE * 1.75f) // Adjusted Probe Distance


// --- End Configuration Constants ---


#define GRID_CONSIDERED (GRID_COLUMNS * ROWS_TO_CONSIDER)
#define DEGREES_PER_COLUMN (FOV_DEGREES / GRID_COLUMNS)

// Global variables
static float safety_grid[GRID_COLUMNS * GRID_ROWS];
static abi_event tensor_output_ev;
static bool is_out_of_bounds = false;

// Function Prototypes (declarations)
static void calculate_vfh_steering(const uint8_t column_ratings[], float *desired_heading_change_deg, float *desired_speed_factor);
static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
static uint8_t increase_nav_heading(float incrementDegrees);
static void get_column_safety_ratings(const float full_grid[], uint8_t column_ratings[]);
static uint8_t get_forward_clearance_rating(const uint8_t column_ratings[]);


/*
 * Callback function to receive the tensor output from the neural network.
 */
static void tensor_output_cb(uint8_t __attribute__((unused)) sender_id, float* tensor)
{
  if (tensor == NULL) {
    PRINT("Tensor_output is NULL\n");
    return;
  }
  int tensor_size = GRID_COLUMNS * GRID_ROWS;
  for (int i = 0; i < tensor_size; i++){
    // Basic validation/clamping if needed: tensor[i] = fmaxf(0.0f, tensor[i]);
    safety_grid[i] = tensor[i];
  }
}

/*
 * Initialisation function
 */
void object_avoider_init(void)
{
  srand(time(NULL));
  AbiBindMsgTENSOR_OUTPUT(TENSOR_OUTPUT_id, &tensor_output_ev, tensor_output_cb);
  PRINT("Object Avoider Initialized. Waiting for safety grid data.\n");
  PRINT("CONFIG: REPULSION_EXPONENT=%.2f, GOAL_STRENGTH=%.1f, OOB_PROBE_DIST=%.2f\n", REPULSION_EXPONENT, GOAL_STRENGTH, OUT_OF_BOUNDS_PROBE_DIST);
}


/*
 * Periodic function, called frequently. Main control logic resides here.
 */
void object_avoider_periodic(void)
{
  if (!autopilot_in_flight()) {
    is_out_of_bounds = false;
    return;
  }

  // 1. Check Boundaries using Drone's Current Position
  struct EnuCoor_f* current_pos_f_ptr = stateGetPositionEnu_f();
  bool drone_is_inside = false;

  if (current_pos_f_ptr != NULL) {
    // Assuming InsideObstacleZone takes floats:
    drone_is_inside = InsideObstacleZone(current_pos_f_ptr->x, current_pos_f_ptr->y);
  } else {
    PRINT("Warning: Cannot get current position to check boundaries! Assuming inside.\n");
    drone_is_inside = true; // Failsafe assumption
  }

  // Update the out-of-bounds state flag
  if (!drone_is_inside && !is_out_of_bounds) {
      PRINT("Drone is Outside Obstacle Zone! Initiating recovery.\n");
      is_out_of_bounds = true;
  } else if (drone_is_inside && is_out_of_bounds) {
      PRINT("Drone Re-entered Obstacle Zone. Resuming normal navigation.\n");
      is_out_of_bounds = false;
  }

  // 2. Process Safety Grid Data (only needed for normal navigation)
  uint8_t column_ratings[GRID_COLUMNS];
  if (!is_out_of_bounds) {
    get_column_safety_ratings(safety_grid, column_ratings);
  }

  float desired_heading_change_deg = 0.0f;
  // desired_speed_factor from VFH is no longer directly used for speed, but calculated for info/potential future use
  float desired_speed_factor_info = 0.0f;


  // 3. Calculate Steering and Speed based on state
  if (is_out_of_bounds) {
      // --- OUT OF BOUNDS LOGIC ---
      desired_heading_change_deg = OUT_OF_BOUNDS_TURN_DEG;

      PRINT("Out of bounds: Turning %.1f deg and probing forward %.2f m.\n", desired_heading_change_deg, OUT_OF_BOUNDS_PROBE_DIST);

      increase_nav_heading(desired_heading_change_deg);

      // ** FIX: Move WP_GOAL forward the increased probe distance **
      moveWaypointForward(WP_GOAL, OUT_OF_BOUNDS_PROBE_DIST); // Probe forward
      if(current_pos_f_ptr != NULL) {
        waypoint_set_alt(WP_GOAL, current_pos_f_ptr->z);
      }

      // Keep other waypoints managed simply
      struct EnuCoor_i traj_coor;
      if (!calculateForwards(&traj_coor, OUT_OF_BOUNDS_PROBE_DIST + WAYPOINT_DISTANCE * 0.5f )) {
          moveWaypoint(WP_TRAJECTORY, &traj_coor);
          if(current_pos_f_ptr != NULL) waypoint_set_alt(WP_TRAJECTORY, current_pos_f_ptr->z);
      }
      struct EnuCoor_i retreat_coor;
      if (!calculateForwards(&retreat_coor, -WAYPOINT_DISTANCE * 0.25f)) {
          moveWaypoint(WP_RETREAT, &retreat_coor);
           if(current_pos_f_ptr != NULL) waypoint_set_alt(WP_RETREAT, current_pos_f_ptr->z);
      }

  } else {
      // --- NORMAL NAVIGATION LOGIC (INSIDE BOUNDS) ---
      // Calculate desired heading using VFH
      calculate_vfh_steering(column_ratings, &desired_heading_change_deg, &desired_speed_factor_info); // Gets heading change and informational speed factor

      // Limit the heading change per step
      Bound(desired_heading_change_deg, -MAX_HEADING_CHANGE_DEG, MAX_HEADING_CHANGE_DEG);

      VERBOSE_PRINT("VFH Steering: HeadingChange=%.1f deg (SpeedFactorInfo=%.2f)\n", desired_heading_change_deg, desired_speed_factor_info);

      increase_nav_heading(desired_heading_change_deg);

      // ** FIX: Calculate speed based *only* on forward clearance **
      uint8_t forward_clearance = get_forward_clearance_rating(column_ratings);
      float clearance_factor = ((float)forward_clearance / MAX_RATING); // 0.0 to 1.0
      // Ensure clearance factor isn't negative if rating somehow exceeds MAX_RATING
      clearance_factor = fmaxf(0.0f, fminf(1.0f, clearance_factor));

      float target_speed = MIN_FORWARD_SPEED + (MAX_FORWARD_SPEED - MIN_FORWARD_SPEED) * clearance_factor;
      Bound(target_speed, MIN_FORWARD_SPEED, MAX_FORWARD_SPEED);

      VERBOSE_PRINT("Forward clearance: %d (Factor=%.2f) -> Target speed: %.2f m/s\n", forward_clearance, clearance_factor, target_speed);

      // Update WP_GOAL distance based on target speed
      float dynamic_wp_dist = fmaxf(WAYPOINT_DISTANCE, target_speed * 1.0f); // e.g., 1 second reaction distance
      Bound(dynamic_wp_dist, WAYPOINT_DISTANCE, WAYPOINT_DISTANCE * 2.5f); // Allow slightly further WP when fast&clear
      VERBOSE_PRINT("Dynamic WP Distance: %.2f m\n", dynamic_wp_dist);

      moveWaypointForward(WP_GOAL, dynamic_wp_dist);
      if(current_pos_f_ptr != NULL) {
        waypoint_set_alt(WP_GOAL, current_pos_f_ptr->z);
      }

      // Update other waypoints relative to current state and desired heading
      struct EnuCoor_i traj_coor;
      if (!calculateForwards(&traj_coor, dynamic_wp_dist + WAYPOINT_DISTANCE * 0.5f)) {
          moveWaypoint(WP_TRAJECTORY, &traj_coor);
          if(current_pos_f_ptr != NULL) waypoint_set_alt(WP_TRAJECTORY, current_pos_f_ptr->z);
      }
      struct EnuCoor_i retreat_coor;
      if (!calculateForwards(&retreat_coor, -WAYPOINT_DISTANCE * 0.5f)) {
          moveWaypoint(WP_RETREAT, &retreat_coor);
           if(current_pos_f_ptr != NULL) waypoint_set_alt(WP_RETREAT, current_pos_f_ptr->z);
      }
  }

  return;
}


/*
 * Calculates the average safety rating for the top ROWS_TO_CONSIDER rows for each column.
 */
void get_column_safety_ratings(const float full_grid[], uint8_t column_ratings[])
{
  if (ROWS_TO_CONSIDER <= 0) {
      for (int c = 0; c < GRID_COLUMNS; c++) column_ratings[c] = 0;
      PRINT("Error: ROWS_TO_CONSIDER is zero or negative.\n");
      return;
  }

  for (int c = 0; c < GRID_COLUMNS; c++) {
      float column_sum = 0.0f;
      for (int r = 0; r < ROWS_TO_CONSIDER; r++) {
          int index = r * GRID_COLUMNS + c;
          if (index < GRID_COLUMNS * GRID_ROWS) { // Basic bounds check
              // Clamp grid values defensively if necessary: fmaxf(0.0f, fminf(MAX_RATING, full_grid[index]))
              column_sum += fmaxf(0.0f, full_grid[index]); // Assuming grid values are >= 0
          }
      }
      float average_rating = column_sum / (float)ROWS_TO_CONSIDER;
      column_ratings[c] = (uint8_t)fminf(MAX_RATING, fmaxf(0.0f, average_rating)); // Clamp result
  }
}

/*
 * Calculates the minimum safety rating in the central columns (forward direction).
 */
uint8_t get_forward_clearance_rating(const uint8_t column_ratings[])
{
  int start_idx = fmaxf(0, CENTRAL_COL_START);
  int end_idx = fminf(GRID_COLUMNS - 1, CENTRAL_COL_END);
  if (start_idx > end_idx) { return 0; }

  uint8_t min_rating = column_ratings[start_idx];
  for (int i = start_idx + 1; i <= end_idx; i++) {
      if (column_ratings[i] < min_rating) {
          min_rating = column_ratings[i];
      }
  }
  return min_rating;
}


/*
 * Core VFH-inspired steering calculation.
 * Output: desired_heading_change_deg, desired_speed_factor (informational)
 */
void calculate_vfh_steering(const uint8_t column_ratings[], float *desired_heading_change_deg, float *desired_speed_factor)
{
    float total_force_x = 0.0f;
    float total_force_y = GOAL_STRENGTH; // Start with forward attraction

    for (int c = 0; c < GRID_COLUMNS; c++) {
        float rating = (float)column_ratings[c];

        // *** TUNED: Use lower exponent for repulsion calculation ***
        float repulsion_magnitude = powf(fmaxf(0.0f, MAX_RATING - rating), REPULSION_EXPONENT);

        if (repulsion_magnitude < FLT_EPSILON) { continue; } // Skip negligible repulsion

        // Optional Threshold: Ignore weak repulsion completely?
        // if (rating > MAX_RATING * 0.8) { continue; } // Example threshold

        float angle_deg = ((float)c - ((float)GRID_COLUMNS / 2.0f) + 0.5f) * DEGREES_PER_COLUMN;
        float angle_rad = RadOfDeg(angle_deg);

        // Force components (pointing away from obstacle direction)
        float force_x = -repulsion_magnitude * sinf(angle_rad);
        float force_y = -repulsion_magnitude * cosf(angle_rad);

        total_force_x += force_x;
        total_force_y += force_y;
    }

    float resultant_angle_rad = atan2f(total_force_x, total_force_y);
    *desired_heading_change_deg = DegOfRad(resultant_angle_rad);

    // Speed factor calculation is now just informational or for potential future use
    *desired_speed_factor = fmaxf(0.0f, cosf(resultant_angle_rad));

    // Verbose print moved to periodic function after bounding heading change
}


/*
* Increases the NAV heading. Normalizes the result to [-pi, pi].
*/
uint8_t increase_nav_heading(float incrementDegrees)
{
  struct FloatEulers* current_eulers_ptr = stateGetNedToBodyEulers_f();
  if (current_eulers_ptr == NULL) {
    PRINT("Error: stateGetNedToBodyEulers_f() returned NULL.\n");
    return true;
  }
  float current_heading_rad = current_eulers_ptr->psi;
  float new_heading_rad = current_heading_rad + RadOfDeg(incrementDegrees);

  // Normalize heading using standard math functions
   while (new_heading_rad > M_PI) new_heading_rad -= 2.0 * M_PI;
   while (new_heading_rad <= -M_PI) new_heading_rad += 2.0 * M_PI;
  // Or use Paparazzi's FLOAT_ANGLE_NORMALIZE(new_heading_rad);

  nav.heading = new_heading_rad; // Assumes nav.heading expects radians

  VERBOSE_PRINT("Current heading: %.1f deg, Increment: %.1f deg, New nav.heading: %.1f deg\n",
                 DegOfRad(current_heading_rad), incrementDegrees, DegOfRad(new_heading_rad));
  return false;
}

/*
* Calculates coordinates 'distanceMeters' ahead based on nav.heading and current position.
*/
uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
{
  float heading_rad = nav.heading;
  struct EnuCoor_f* current_pos_f_ptr = stateGetPositionEnu_f();

  if (current_pos_f_ptr == NULL) {
      PRINT("Error: stateGetPositionEnu_f() returned NULL pointer in calculateForwards.\n");
      return true;
  }

  float new_x_f = current_pos_f_ptr->x + sinf(heading_rad) * distanceMeters;
  float new_y_f = current_pos_f_ptr->y + cosf(heading_rad) * distanceMeters;

  new_coor->x = POS_BFP_OF_REAL(new_x_f);
  new_coor->y = POS_BFP_OF_REAL(new_y_f);

  return false;
}

/*
* Moves waypoint 'waypoint' to the integer coordinates 'new_coor'.
*/
uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
{
  waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
  return false;
}

/*
* Sets waypoint 'waypoint' a certain distance ahead based on nav.heading.
*/
uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
{
  struct EnuCoor_i new_coor;
  if (calculateForwards(&new_coor, distanceMeters)) {
      PRINT("Error calculating forward position for WP %d.\n", waypoint);
      return true;
  }
  moveWaypoint(waypoint, &new_coor);
  return false;
}