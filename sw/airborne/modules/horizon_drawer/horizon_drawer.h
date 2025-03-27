#ifndef HORIZON_DRAWER_H
#define HORIZON_DRAWER_H

extern int middle_danger_zone;
extern int min_safe_dist;
extern float skip_percentage;
extern int num_neighbors;
extern int EDGE_THRESH;

void horizon_drawer_init(void);
void horizon_drawer_periodic(void);

#endif

