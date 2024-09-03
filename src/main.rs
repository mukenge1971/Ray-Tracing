//mod utils;
//use crate::utils::vector3::Vecto;
use std::fs::File;
use std::io::Write;
use std::ops::{Add, Div, Mul, Sub};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vecto {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub struct Ray {
    pub origin: Vecto,
    pub direction: Vecto,
}

pub struct Light {
    pub position: Vecto,
    pub intensity: f64,
    pub color: Vecto,
}

impl Mul<f64> for Vecto {
    type Output = Vecto;

    fn mul(self, num: f64) -> Vecto {
        Vecto {
            x: num * self.x,
            y: num * self.y,
            z: num * self.z,
        }
    }
}
impl Mul<Vecto> for f64 {
    type Output = Vecto;

    fn mul(self, vecto: Vecto) -> Vecto {
        Vecto {
            x: vecto.x * self,
            y: vecto.y * self,
            z: vecto.z * self,
        }
    }
}

impl Ray {
    pub fn new(origin: Vecto, direction: Vecto) -> Self {
        Ray { origin, direction }
    }
    pub fn at(&self, t: f64) -> Vecto {
        self.origin + (t * self.direction)
    } //self.origin.add(t*self.direction)

    pub fn come_from_outside(&self, normal: Vecto) -> bool {
        self.direction * normal < 0.0
    }
}
//use std::{ops::{Add, Sub, Mul, Div}, fmt::Display};

pub struct Sphere {
    pub center: Vecto,
    pub radius: f64,
    pub color: Vecto, // RGB color
}

pub struct Plane {
    pub point: Vecto,
    pub normal: Vecto,
    pub color: Vecto, // RGB color
}

pub struct Cube {
    pub min: Vecto,
    pub max: Vecto,
    pub color: Vecto, // RGB color
}

impl Add<Vecto> for Vecto {
    type Output = Vecto;

    fn add(self, other: Self) -> Self::Output {
        Vecto {
            x: self.x + other.x,
            y: self.y + other.y,
            z: self.z + other.z,
        }
    }
}
impl Sub<Vecto> for Vecto {
    type Output = Vecto;

    fn sub(self, other: Self) -> Self::Output {
        Vecto {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }
}

impl Mul<Vecto> for Vecto {
    type Output = f64;

    fn mul(self, rhs: Vecto) -> f64 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
}
impl Div<f64> for Vecto {
    type Output = Vecto;

    fn div(self, numb: f64) -> Self::Output {
        Vecto {
            x: self.x / numb,
            y: self.y / numb,
            z: self.z / numb,
        }
    }
}
impl Vecto {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vecto { x, y, z }
    }
    pub fn cross(&self, other: &Vecto) -> Vecto {
        Vecto {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    // Méthode pour calculer la longueur du vecteur
    pub fn length(&self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }

    // Méthode pour obtenir le vecteur unitaire (de longueur 1) dans la même direction
    pub fn unit_vector(&self) -> Vecto {
        let length = self.length();
        Vecto {
            x: self.x / length,
            y: self.y / length,
            z: self.z / length,
        }
    }
    pub fn multiply_matrix(&self, matrix: &[[f64; 3]; 3]) -> Self {
        let x = self.x * matrix[0][0] + self.y * matrix[0][1] + self.z * matrix[0][2];
        let y = self.x * matrix[1][0] + self.y * matrix[1][1] + self.z * matrix[1][2];
        let z = self.x * matrix[2][0] + self.y * matrix[2][1] + self.z * matrix[2][2];

        Self { x, y, z }
    }
}

pub struct Cylinder {
    pub base_center: Vecto,
    pub axis_direction: Vecto,
    pub radius: f64,
    pub height: f64,
    pub color: Vecto, // RGB color
}

fn intersect_sphere(ray: &Ray, sphere: &Sphere) -> Option<f64> {
    let oc = ray.origin - sphere.center;
    let a = ray.direction * ray.direction;
    let b = oc * (ray.direction);
    let c = oc * oc - sphere.radius * sphere.radius;
    
    let discriminant = b * b - a * c;
    //print!("{:?} ", discriminant);

    if discriminant > 0.0 {
        let sqrt_discriminant = discriminant.sqrt();
        let t1 = (-b - sqrt_discriminant) / a;
        let t2 = (-b + sqrt_discriminant) / a;

        return if t1 >= 0.0 {
            Some(t1)
        } else if t2 >= 0.0 {
            Some(t2)
        } else {
            None
        }
    }
    None
}
// impl Vecto {
//     // Méthode pour soustraire un autre Vecto à ce Vecto.
//     fn sub(&self, other: &Vecto) -> Vecto {
//         Vecto {
//             x: self.x - other.x,
//             y: self.y - other.y,
//             z: self.z - other.z,
//         }
//     }
// }

fn intersect_plane(ray: &Ray, plane: &Plane) -> Option<f64> {
    let denom = ray.direction.unit_vector() * (plane.normal);

    if denom.abs() > 1e-6 {
        let t = (plane.point - ray.origin) * (plane.normal) / denom;

        if t >= 0.0 {
            return  Some(t)
        }
    }
    None
}

fn intersect_cylinder(ray: &Ray, cylinder: &Cylinder) -> Option<f64> {
    // Calculate the coefficients of the quadratic equation
    let a = ray.direction.x * ray.direction.x + ray.direction.z * ray.direction.z;
    let b = 2.0
        * ((ray.origin.x - cylinder.base_center.x) * ray.direction.x
            + (ray.origin.z - cylinder.base_center.z) * ray.direction.z);
    let c = (ray.origin.x - cylinder.base_center.x) * (ray.origin.x - cylinder.base_center.x)
        + (ray.origin.z - cylinder.base_center.z) * (ray.origin.z - cylinder.base_center.z)
        - cylinder.radius * cylinder.radius;

    // Calculate the discriminant
    let discriminant = b * b - 4.0 * a * c;

    // If the discriminant is negative, there are no intersections
    if discriminant < 0.0 {
        return None;
    }

    // Calculate the two possible values of t
    let sqrt_discriminant = discriminant.sqrt();
    let t1 = (-b - sqrt_discriminant) / (2.0 * a);
    let t2 = (-b + sqrt_discriminant) / (2.0 * a);

    // Check the Y values at the intersection points
    let y1 = ray.origin.y + t1 * ray.direction.y;
    let y2 = ray.origin.y + t2 * ray.direction.y;

    let h = cylinder.height;

    if (y1 >= cylinder.base_center.y - h && y1 <= cylinder.base_center.y) && t1 >= 0.0 {
        return Some(t1);
    } else if (y2 >= cylinder.base_center.y - h && y2 <= cylinder.base_center.y) && t2 >= 0.0 {
        return Some(t2);
    }

    None
}

fn intersect_cube(ray: &Ray, cube: &Cube) -> Option<f64> {
    let mut t_min = (cube.min.x - ray.origin.x) / ray.direction.x;
    let mut t_max = (cube.max.x - ray.origin.x) / ray.direction.x;

    let mut t_ymin = (cube.min.y - ray.origin.y) / ray.direction.y;
    let mut t_ymax = (cube.max.y - ray.origin.y) / ray.direction.y;

    if t_ymin > t_ymax {
        std::mem::swap(&mut t_ymin, &mut t_ymax);
    }

    if t_min > t_ymax || t_ymin > t_max {
        return None;
    }

    if t_ymin > t_min {
        t_min = t_ymin;
    }

    if t_ymax < t_max {
        t_max = t_ymax;
    }

    let t_zmin = (cube.min.z - ray.origin.z) / ray.direction.z;
    let t_zmax = (cube.max.z - ray.origin.z) / ray.direction.z;

    if t_min > t_zmax || t_zmin > t_max {
        return None;
    }

    Some(f64::max(t_min, t_zmin))
}

//use crate::vector::Vecto; // Assuming you have a Vecto struct for 3D vectors

pub struct Camera {
    pub origin: Vecto,            // Position of the camera
    pub lower_left_corner: Vecto, // Lower-left corner of the viewport
    pub horizontal: Vecto,        // Horizontal dimension of the viewport
    pub vertical: Vecto,          // Vertical dimension of the viewport
}

impl Camera {
    // Constructor for creating a new camera
    pub fn new(
        look_from: Vecto,
        look_at: Vecto,
        up: Vecto,
        vertical_fov: f64,
        aspect_ratio: f64,
    ) -> Camera {
        let theta = vertical_fov.to_radians();
        let h = (theta / 2.0).tan();
        let viewport_height = 2.0 * h;
        let viewport_width = aspect_ratio * viewport_height;
        let w = (look_from - look_at).unit_vector();
        let u = up.cross(&w).unit_vector();
        let v = w.cross(&u);

        let origin = look_from;
        let horizontal = viewport_width * u;
        let vertical = viewport_height * v;
        let lower_left_corner = origin - horizontal / 2.0 - vertical / 2.0 - w;

        Camera {
            origin,
            lower_left_corner,
            horizontal,
            vertical,
        }
    }

    // Function to calculate the ray direction from camera for given u and v coordinates
    pub fn get_ray(&self, u: f64, v: f64) -> Ray {
        let direction =
            self.lower_left_corner + u * self.horizontal + v * self.vertical - self.origin;
        Ray::new(self.origin, direction)
    }

    pub fn translate(&mut self, delta: Vecto) {
        self.origin.x += delta.x;
        self.origin.y += delta.y;
        self.origin.z += delta.z;
    }
    pub fn rotate_yaw(&mut self, angle: f64) {
        let cos_theta = angle.cos();
        let sin_theta = angle.sin();

        // Rotate the lower_left_corner around the y-axis
        let x = self.lower_left_corner.x;
        let z = self.lower_left_corner.z;
        self.lower_left_corner.x = cos_theta * x + sin_theta * z;
        self.lower_left_corner.z = -sin_theta * x + cos_theta * z;

        // Rotate the horizontal vector around the y-axis
        let x = self.horizontal.x;
        let z = self.horizontal.z;
        self.horizontal.x = cos_theta * x + sin_theta * z;
        self.horizontal.z = -sin_theta * x + cos_theta * z;

        // Rotate the vertical vector around the y-axis (though for many cameras, this might be unnecessary if vertical is always aligned with global up)
        let x = self.vertical.x;
        let z = self.vertical.z;
        self.vertical.x = cos_theta * x + sin_theta * z;
        self.vertical.z = -sin_theta * x + cos_theta * z;
    }
    
    
}

trait Intersectable {
    fn intersect(&self, ray: &Ray) -> Option<f64>;
    fn color(&self) -> Vecto;
    fn surface_normal(&self, hit_point: &Vecto) -> Vecto;
}

impl Intersectable for Sphere {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        intersect_sphere(ray, self)
    }

    fn color(&self) -> Vecto {
        self.color
    }

    fn surface_normal(&self, hit_point: &Vecto) -> Vecto {
        (self.center - *hit_point).unit_vector()
    }
}

impl Intersectable for Cube {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        intersect_cube(ray, self)
    }

    fn color(&self) -> Vecto {
        self.color
    }

    fn surface_normal(&self, hit_point: &Vecto) -> Vecto {
        let epsilon = 1e-6; // Une petite valeur pour gérer les imprécisions des calculs flottants

        if (hit_point.x - self.min.x).abs() < epsilon {
            return Vecto {
                x: 1.0,
                y: 0.0,
                z: 0.0,
            }; // Face gauche
        }
        if (hit_point.x - self.max.x).abs() < epsilon {
            return Vecto {
                x: -1.0,
                y: 0.0,
                z: 0.0,
            }; // Face droite
        }

        if (hit_point.y - self.min.y).abs() < epsilon {
            return Vecto {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            }; // Face inférieure
        }
        if (hit_point.y - self.max.y).abs() < epsilon {
            return Vecto {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            }; // Face supérieure
        }

        if (hit_point.z - self.min.z).abs() < epsilon {
            return Vecto {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            }; // Face arrière
        }
        if (hit_point.z - self.max.z).abs() < epsilon {
            return Vecto {
                x: 0.0,
                y: 0.0,
                z: -1.0,
            }; // Face avant
        }

        // En théorie, on ne devrait jamais atteindre ce point si le point de collision est réellement sur le cube.
        // Mais pour être sûr, on retourne une normale par défaut.
        return Vecto {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        };
    }
}

impl Intersectable for Cylinder {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        intersect_cylinder(ray, self)
    }

    fn color(&self) -> Vecto {
        self.color
    }

    fn surface_normal(&self, hit_point: &Vecto) -> Vecto {
        // Si le hit_point est proche de la base inférieure
        if (hit_point.y - self.base_center.y).abs() < 1e-6 {
            return Vecto {
                x: 0.0,
                y: -1.0,
                z: 0.0,
            };
        }

        // Si le hit_point est proche de la base supérieure
        if (hit_point.y - (self.base_center.y + self.height)).abs() < 1e-6 {
            return Vecto {
                x: 0.0,
                y: 1.0,
                z: 0.0,
            };
        }

        // Si le rayon frappe le côté
        let base_to_hit = Vecto {
            x: self.base_center.x - hit_point.x,
            y: 0.0,
            z: self.base_center.z - hit_point.z,
        };
        base_to_hit.unit_vector()
    }
}

impl Intersectable for Plane {
    fn intersect(&self, ray: &Ray) -> Option<f64> {
        intersect_plane(ray, self)
    }

    fn color(&self) -> Vecto {
        self.color
    }

    fn surface_normal(&self, _hit_point: &Vecto) -> Vecto {
        self.normal
    }
}


fn get_color(ray: &Ray, objects: &[&dyn Intersectable], lights: &[Light]) -> Vecto {
    let mut closest_distance = f64::INFINITY;
    let mut final_color = Vecto::new(0.6, 0.6, 0.9);  // default black color

    for object in objects {
        if let Some(t) = object.intersect(ray) {
            if t < closest_distance {
                closest_distance = t;

                // Calculate the point of intersection
                let point_of_intersection = ray.at(t);

                // Initialize the shaded color as black
                let mut shaded_color = Vecto::new(0., 0., 0.);
                for light in lights.iter() {
                    // Compute direction to the light
                    let light_direction = (point_of_intersection - light.position).unit_vector();
                    // Compute surface normal
                    // Here we're approximating the normal by simply taking the direction from the object's center to the point of intersection. This works for spheres.
                    let surface_normal = object.surface_normal(&point_of_intersection);
                    // // Lambertian reflection
                    let lambert = surface_normal * light_direction;
                    let lambert = lambert.max(0.);//if lambert < 0.0 { 0.0 } else { lambert };
                    // Add contribution of this light to the shaded color
                    //shaded_color = shaded_color + (object.color() +light.color) * lambert * light.intensity;
                    let light_effect = Vecto::new(
                        object.color().x * light.color.x * lambert * light.intensity,
                        object.color().y * light.color.y * lambert * light.intensity,
                        object.color().z * light.color.z * lambert * light.intensity
                    );

                    // Manual addition
                    shaded_color = Vecto::new(
                        shaded_color.x + light_effect.x,
                        shaded_color.y + light_effect.y,
                        shaded_color.z + light_effect.z
                    );

                }
                // Combine the colors

                final_color =  shaded_color;
            }
        }
    }

    final_color
}

const WIDTH: usize = 800; // Largeur de l'image
const HEIGHT: usize = 600; // Hauteur de l'image

fn main() {
    let aspect_ratio = WIDTH as f64 / HEIGHT as f64; //= 4./3.;//16.0/9.;
                                                       // let width = 800;
                                                       // let height = (800.0 / aspect_ratio) as usize;
    let mut framebuffer: Vec<Vecto> = Vec::with_capacity(WIDTH * HEIGHT); // Initialisé avec une couleur noire

    for _ in 0..HEIGHT {
        for _ in 0..WIDTH {
            framebuffer.push(Vecto::new(0., 0., 0.));
        }
    }

    // Définition de la sphère
    let sphere = Sphere {
        center: Vecto {
            x: -1.0,
            y: 0.0,
            z: -5.0,
        },
        radius: 0.75,
        color: Vecto {
            x: 0.4,
            y: 0.0,
            z: 0.0,
        }, // Rouge
    };
    let sphere2 = Sphere {
        center: Vecto {
            x: 2.0,
            y: 0.0,
            z: -5.0,
        },
        radius: 0.5,
        color: Vecto {
            x: 0.0,
            y: 0.2,
            z: 0.0,
        }, 
    };
    // let sphere3 = Sphere {
    //     center: Vecto {
    //         x: 2.0,
    //         y: 0.0,
    //         z: -5.0,
    //     },
    //     radius: 0.5,
    //     color: Vecto {
    //         x: 0.0,
    //         y: 0.2,
    //         z: 0.0,
    //     }, 
    // };
    // let sphere4 = Sphere {
    //     center: Vecto {
    //         x: 3.0,
    //         y: 0.0,
    //         z: -5.0,
    //     },
    //     radius: 0.5,
    //     color: Vecto {
    //         x: 0.0,
    //         y: 0.2,
    //         z: 0.0,
    //     }, 
    // };
    // let sphere5 = Sphere {
    //     center: Vecto {
    //         x: 4.0,
    //         y: 0.0,
    //         z: -5.0,
    //     },
    //     radius: 0.5,
    //     color: Vecto {
    //         x: 0.0,
    //         y: 0.2,
    //         z: 0.0,
    //     }, 
    // };
    let cube: Cube = Cube {
        min: Vecto {
            x: 2.,
            y: 1.,
            z: -6.,
        },
        max: Vecto {
            x: 4.,
            y: 3.,
            z: -8.,
        },
        color: Vecto {
            x: 0.,
            y: 0.4,
            z: 0.6,
        },
    };
    let cylinder: Cylinder = Cylinder {
        base_center: Vecto {
            x: 0.,
            y: 1.,
            z: -6.,
        },
        axis_direction: Vecto {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        },
        radius: 1.,
        height: 2.0,
        color: Vecto {
            x: 0.,
            y: 0.,
            z: 0.5,
        },
    };
    let flat_plane: Plane = Plane {
        point: Vecto {
            x: 0.,
            y: 3.,
            z: 0.,
        },
        normal: Vecto {
            x: 0.,
            y: 1.,
            z: 0.,
        },
        color: Vecto {
            x: 0.3,
            y: 0.3,
            z: 0.4,
        },
    };

    // Définition de la caméra
    let mut camera = Camera::new(
        Vecto {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }, // Position de la caméra
        Vecto {
            x: 0.0,
            y: 0.0,
            z: -1.0,
        }, // Point vers lequel la caméra regarde
        Vecto {
            x: 0.0,
            y: 1.0,
            z: 0.0,
        }, // Direction "haut" de la caméra
        90.0, // FOV vertical
        aspect_ratio,
    );
    //camera.translate(Vecto { x: 1., y: 0., z: 1. });
    let lights = [Light {
        position: Vecto {
            x: 0.,
            y: -2.,
            z: 0.,
        },
        intensity: 4.,
        color: Vecto {
            x: 0.5,
            y: 0.5,
            z: 0.5,
        },
    }];
     camera.rotate_yaw((5_f64).to_radians());

    // Dimensions de l'écran

    let mut cursor: usize = 0;
    // Raytracing pour chaque pixel
    for j in 0..HEIGHT {
        for i in 0..WIDTH {
            let u = i as f64 / WIDTH as f64;
            let v = j as f64 / HEIGHT as f64;
            let ray = camera.get_ray(u, v);
            // let color = get_color(&ray, &[&sphere,&sphere2,&sphere3, &sphere4, &sphere5], &lights);//&flat_plane,&cube,&cylinder],&lights);
            let color = get_color(&ray, &[&sphere,&sphere2,&cube,&cylinder,&flat_plane],&lights);

            framebuffer[cursor] = color;
            cursor += 1;
        }
    }
    let image_count = std::env::args().nth(1) .expect("Pas d'argument !!"); 
    let mut output = File::create(format!("result_{}.ppm", image_count)).unwrap();
    writeln!(output, "P3").unwrap();
    writeln!(output, "{} {}", WIDTH, HEIGHT).unwrap();
    writeln!(output, "{}", 255).unwrap();

    for color in framebuffer.iter() {
        writeln!(
            output,
            "{} {} {}",
            (color.x * 255.) as u8,
            (color.y * 255.) as u8,
            (color.z * 255.) as u8
        )
        .unwrap();
    }
    print!("Completely done. Image save as \"result_{}.ppm\"\n", image_count)
}
