// src/vector/mod.rs

pub struct Vecto {
    x: f64,
    y: f64,
    z: f64,
}

impl Vecto {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Vecto { x, y, z }
    }
}
