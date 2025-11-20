from dataclasses import dataclass
from pyroki.collision._geometry import CollGeom, HalfSpace, Sphere, Capsule, Box
from jaxtyping import Float, Array
@dataclass
class Obstacle:
    coll_geom: CollGeom
    @classmethod
    def from_sphere(cls, center: Float[Array, "*batch 3"], radius: Float[Array, "*batch"]) -> "Obstacle":
        """Create an obstacle from a batch of spheres.
        
        Args:
            center: The center of the batch of spheres.
            radius: The radius of the batch of spheres.
            
        Returns:
            An obstacle.
        """
        return cls(Sphere.from_center_and_radius(center, radius))
