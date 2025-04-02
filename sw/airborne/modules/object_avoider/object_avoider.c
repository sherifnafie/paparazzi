// This file is the main file for the object avoider module. It is responsible for avoiding obstacles in the drone's path.
// The module uses a neural network to detect obstacles and then uses the detected obstacles to avoid them.
// The module uses a state machine to determine the drone's current state and then takes appropriate action to avoid obstacles.


 #include "modules/object_avoider/object_avoider.h"
 #include "firmwares/rotorcraft/navigation.h"
 #include "generated/airframe.h"
 #include "state.h"
 #include "modules/core/abi.h"
 #include <time.h>
 #include <stdio.h>
  
 #define NAV_C // needed to get the nav functions like Inside...
 #include "generated/flight_plan.h"
  
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
 
 #define COLUMNS_TO_CONSIDER 32
 #define ROWS_TO_CONSIDER 9
 #define GRID_CONSIDERED (COLUMNS_TO_CONSIDER * ROWS_TO_CONSIDER)
 #define INCREMENT_STEP 6 // FOV/columns = 180/30 = 6, (32-borders=30)
  
 static uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters);
 static uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters);
 static uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor);
 static uint8_t increase_nav_heading(float incrementDegrees);
 static uint8_t chooseRandomIncrementAvoidance(void);
 void get_column_safety_ratings(uint8_t column_ratings[]);
 uint8_t get_safety_rating(uint8_t column_ratings[]);
 int8_t get_heading_increment(uint8_t column_ratings[]);
  
 enum navigation_state_t {
   SAFE,
   OBSTACLE_FOUND,
   SEARCH_FOR_SAFE_HEADING,
   OUT_OF_BOUNDS
   };
 
 // define settings
 //float oa_color_count_frac = 0.18f;
 
 // define and initialise global variables
 enum navigation_state_t navigation_state = SEARCH_FOR_SAFE_HEADING;
 uint8_t safety_rating = 0;              // starting safety rating
 uint8_t safety_minimum = 70;            // minimum safety rating to be considered safe 
 int16_t obstacle_free_confidence = 0;   // a measure of how certain we are that the way ahead is safe.
//  int8_t heading_increment = 10;          // heading angle increment [deg]
 float maxDistance = 2.25;               // max waypoint displacement [m]
 
 const int16_t max_trajectory_confidence = 5; // number of consecutive negative object detections to be sure we are obstacle free
 
 
 float safety_grid[GRID_CONSIDERED];                    // tensor output from the neural network
 
 static abi_event tensor_output_ev;
 static void tensor_output_cb(uint8_t __attribute__((unused)) sender_id, float* tensor)
 {
   if (tensor == NULL) { 
     printf("Tensor_output is NULL\n"); 
     return; 
   }
   // copy tensor output to local variable safety_grid
   for (int i = 0; i < GRID_CONSIDERED; i++){
     safety_grid[i] = tensor[i];
   } 
 }
  
  /*
   * Initialisation function, setting the colour filter, random seed and heading_increment
   */
 void object_avoider_init(void)
 {
   // Initialise random values
   srand(time(NULL));
   //chooseRandomIncrementAvoidance();
  
   // bind our colorfilter callbacks to receive the color filter outputs
   AbiBindMsgTENSOR_OUTPUT(TENSOR_OUTPUT_id, &tensor_output_ev, tensor_output_cb);
 }
 
 
 
 void object_avoider_periodic(void)
 {
   // // print model ouput
   // printf("safety grid: \n");
 
   // // loop array
   // for (int i = 0; i < 384; i++) {
   //   printf("%f ", safety_grid[i]);
 
   //   if ((i + 1) % 32 == 0) {
   //     printf("\n");  // Print new line after every 16th element
   //   }
   // }
   uint8_t column_ratings[COLUMNS_TO_CONSIDER]; // safety ratings for each column in the safety grid
   get_column_safety_ratings(column_ratings);
   int8_t heading_increment = get_heading_increment(column_ratings);
   safety_rating = get_safety_rating(column_ratings);

  //  printf("Column safety ratings: \n");
  //  for (int i = 0; i < COLUMNS_TO_CONSIDER; i++) {
  //    printf("%d ", column_ratings[i]);
  //  }
   printf("Heading increment: %d\n", heading_increment);
   printf("Safety rating: %d\n", safety_rating);
 
   // only evaluate our state machine if we are flying
   if(!autopilot_in_flight()){
     return;
   }
 
   // update our safe confidence using color threshold
   if(safety_rating > safety_minimum){
     obstacle_free_confidence++;
   } else {
     obstacle_free_confidence -= 2;  // be more cautious with positive obstacle detections
   }
   printf("obstacle_free_confidence: %d\n", obstacle_free_confidence);
 
   // bound obstacle_free_confidence
   Bound(obstacle_free_confidence, 0, max_trajectory_confidence);
 
   float moveDistance = fminf(maxDistance, 0.2f * obstacle_free_confidence);

  printf("Current state: %s\n", 
  navigation_state == SAFE ? "SAFE" :
  navigation_state == OBSTACLE_FOUND ? "OBSTACLE_FOUND" :
  navigation_state == SEARCH_FOR_SAFE_HEADING ? "SEARCH_FOR_SAFE_HEADING" :
  navigation_state == OUT_OF_BOUNDS ? "OUT_OF_BOUNDS" : "UNKNOWN");

   switch (navigation_state){
     case SAFE:
       // Move waypoint forward
       moveWaypointForward(WP_TRAJECTORY, 1.0f * moveDistance);
       if (!InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
         navigation_state = OUT_OF_BOUNDS;
       } else if (obstacle_free_confidence == 0){
         navigation_state = OBSTACLE_FOUND;
       } else {
         moveWaypointForward(WP_GOAL, moveDistance);
         moveWaypointForward(WP_RETREAT, -1.0f * moveDistance);
       }
 
       break;
     case OBSTACLE_FOUND:
       // stop
       waypoint_move_here_2d(WP_GOAL);
       waypoint_move_here_2d(WP_RETREAT);
       waypoint_move_here_2d(WP_TRAJECTORY);
 
       // randomly select new search direction
       //chooseRandomIncrementAvoidance();
 
       navigation_state = SEARCH_FOR_SAFE_HEADING;
 
       break;
     case SEARCH_FOR_SAFE_HEADING:
       increase_nav_heading(heading_increment);
       navigation_state = SAFE;
       // make sure we have a couple of good readings before declaring the way safe
      //  if (obstacle_free_confidence >= 2){
      //    navigation_state = SAFE;
      //  }
       break;
     case OUT_OF_BOUNDS:
       increase_nav_heading(35.0f);
       moveWaypointForward(WP_TRAJECTORY, 1.5f);
       moveWaypointForward(WP_RETREAT, -1.0f);
 
       if (InsideObstacleZone(WaypointX(WP_TRAJECTORY),WaypointY(WP_TRAJECTORY))){
         // add offset to head back into arena
         // increase_nav_heading(heading_increment);
 
         // reset safe counter
         obstacle_free_confidence = 0;
 
         // ensure direction is safe before continuing
         navigation_state = SEARCH_FOR_SAFE_HEADING;
       }
       break;
     default:
       break;
   }
   return;
 }
 
 
  /*
  * Increases the NAV heading. Assumes heading is an INT32_ANGLE. It is bound in this function.
  */
 uint8_t increase_nav_heading(float incrementDegrees)
 {
   float new_heading = stateGetNedToBodyEulers_f()->psi + RadOfDeg(incrementDegrees);
 
   // normalize heading to [-pi, pi]
   FLOAT_ANGLE_NORMALIZE(new_heading);
 
   // set heading, declared in firmwares/rotorcraft/navigation.h
   nav.heading = new_heading;
 
   VERBOSE_PRINT("Increasing heading to %f\n", DegOfRad(new_heading));
   return false;
 }
 
 /*
  * Calculates coordinates of distance forward and sets waypoint 'waypoint' to those coordinates
  */
 uint8_t moveWaypointForward(uint8_t waypoint, float distanceMeters)
 {
   struct EnuCoor_i new_coor;
   calculateForwards(&new_coor, distanceMeters);
   moveWaypoint(waypoint, &new_coor);
   return false;
 }
 
 /*
  * Calculates coordinates of a distance of 'distanceMeters' forward w.r.t. current position and heading
  */
 uint8_t calculateForwards(struct EnuCoor_i *new_coor, float distanceMeters)
 {
   float heading  = stateGetNedToBodyEulers_f()->psi;
 
   // Now determine where to place the waypoint you want to go to
   new_coor->x = stateGetPositionEnu_i()->x + POS_BFP_OF_REAL(sinf(heading) * (distanceMeters));
   new_coor->y = stateGetPositionEnu_i()->y + POS_BFP_OF_REAL(cosf(heading) * (distanceMeters));
   //VERBOSE_PRINT("Calculated %f m forward position. x: %f  y: %f based on pos(%f, %f) and heading(%f)\n", distanceMeters,	
                //  POS_FLOAT_OF_BFP(new_coor->x), POS_FLOAT_OF_BFP(new_coor->y),
                //  stateGetPositionEnu_f()->x, stateGetPositionEnu_f()->y, DegOfRad(heading));
   return false;
 }
 
 /*
  * Sets waypoint 'waypoint' to the coordinates of 'new_coor'
  */
 uint8_t moveWaypoint(uint8_t waypoint, struct EnuCoor_i *new_coor)
 {
  //  VERBOSE_PRINT("Moving waypoint %d to x:%f y:%f\n", waypoint, POS_FLOAT_OF_BFP(new_coor->x),
  //                POS_FLOAT_OF_BFP(new_coor->y));
   waypoint_move_xy_i(waypoint, new_coor->x, new_coor->y);
   return false;
 }
 
 /*
  * Sets the variable 'heading_increment' randomly positive/negative
  */
//  uint8_t chooseRandomIncrementAvoidance(void)
//  {
//    // Randomly choose CW or CCW avoiding direction
//    if (rand() % 2 == 0) {
//      heading_increment = 10.f;
//      VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
//    } else {
//      heading_increment = -10.f;
//      VERBOSE_PRINT("Set avoidance increment to: %f\n", heading_increment);
//    }
//    return false;
//  }
 
 
 void get_column_safety_ratings(uint8_t column_ratings[])
 {
   // Ensure we have rows to consider to avoid division by zero
   if (ROWS_TO_CONSIDER <= 0) {
     // Handle error case: maybe set all ratings to 0 or a specific error value
     for (int c = 0; c < COLUMNS_TO_CONSIDER; c++) {
       column_ratings[c] = 0; // Example: Default to 0 if no rows are considered
     }
     // Optionally print an error message
     // fprintf(stderr, "Error: ROWS_TO_CONSIDER is zero or negative.\n");
     return;
   }
 
   // Iterate over each column (0 to 31)
   for (int c = 0; c < COLUMNS_TO_CONSIDER; c++) {
     float column_sum = 0.0f;
 
     // Iterate over the rows we need to consider for this column (0 to 8)
     for (int r = 0; r < ROWS_TO_CONSIDER; r++) {
       // Calculate the index in the 1D safety_grid array
       // Assuming row-major order: index = row * width + column
       int index = r * COLUMNS_TO_CONSIDER + c;
 
       // Add the grid cell value to the column's sum
       column_sum += safety_grid[index];
     }
 
     // Calculate the average safety rating for the current column
     float average_rating = column_sum / (float)ROWS_TO_CONSIDER;
 
     // Store the result (cast to int16_t) in the output array
     column_ratings[c] = (uint8_t)average_rating;
 
     // Optional: Print the rating for the current column for debugging
     // printf("Column %d Safety Rating: %d\n", c, column_ratings[c]);
   }
   // The column_ratings array now holds the 32 safety ratings.
 }
 
 uint8_t get_safety_rating(uint8_t column_ratings[])
 {
   const int start_index = 13;
   const int end_index = 18;
 
   uint8_t min_rating = column_ratings[start_index];
 
   // Iterate through the rest of the range (from start_index + 1 up to end_index)
   for (int i = start_index + 1; i <= end_index; i++) {
       // If the current rating is higher than the max found so far, update max
       if (column_ratings[i] < min_rating) {
           min_rating = column_ratings[i];
       }
   }
 
   // Return the highest value found in the specific range
   return min_rating;
 }

 int8_t get_heading_increment(uint8_t column_ratings[]) 
 {
  uint8_t max_rating = 0;
  uint8_t max_index = 1;
  
  // Iterate through the range 1 to 30 (ignoring columns 0 and 31)
  for (int i = 1; i <= 30; i++) {
      // Compute the sum of the column's rating and its neighbors
      uint8_t current_rating = column_ratings[i] + column_ratings[i - 1] + column_ratings[i + 1];
      
      // Check if this is the highest rating found
      if (current_rating > max_rating) {
          max_rating = current_rating;
          max_index = i;
      }
  }
  
  // Compute the heading increment (ensure it stays within -90 to 90)
  int8_t heading_increment = (int8_t)((INCREMENT_STEP / 2) * ((max_index * 2) - 31));
  
  return heading_increment;
}