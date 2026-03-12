use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SiteType {
    Atop,
    Hollow,
    Bridge,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdsorbateType {
    Carbon,
    Hydrogen,
    None,
}

#[derive(Debug, Clone)]
pub struct Site{
    pub position: [f64; 3],
    pub site_type: SiteType,
    pub adsorbate_type: AdsorbateType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SurfaceGeometry {
    pub metal: String,
    pub facet: String,
    pub lattice_constant: f64,
    pub dimensions: [usize; 2],
    pub periodic: [bool; 2],
}

impl Default for SurfaceGeometry {
    fn default() -> Self {
        Self {
            metal: "Pt".to_string(),
            facet: "111".to_string(),
            lattice_constant: 3.92,
            dimensions: [10, 10],
            periodic: [true, true],
        }
    }
}