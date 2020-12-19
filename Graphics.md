# Computer Graphics
Computer Graphics is part of computer science that treats visualization and simulation. 

### Ellipsoidal path connections for time-gated rendering
<https://dl.acm.org/doi/10.1145/3306346.3323016>

Time gated rendering is one of rendering method, where the photon/rays are filtered by time they traveled. 
The problem here is, when time gate has narrow accepting interval, the probability of accepting ray is too small.
This makes rendering inefficient, and more if the accepting interval is small.
The technique used in this paper is called ellipsoidal path connection. 
The idea comes from that ellipsoid preserves sum of length from two point.
Given travel time t, compute length of ray l, then choose some of points that this ray be reflected.
Suppose sum of length of path is smaller than l, then from two consecutive points, create ellipsoid using two points as focus, and find intersection with surface.
Using that point as path connection gives ray of length l.
To normalize weight, it uses importance sampling, however not on ellipsoidal connection weight, but classical ray tracing weight.
The normalization is proved by automatic differentiation.
