use std::f64::consts::E;
use serde::{Serialize, Deserialize, Serializer, Deserializer};

#[derive(Clone, Copy, Debug)]
pub struct Activation {
    pub function: fn(&f64) -> f64,
    pub derivative: fn(&f64) -> f64,
}

pub const SIGMOID: Activation = Activation {
    function: |x| 1.0 / (1.0 + E.powf(-x)),
    derivative: |x| x * (1.0 - x),
};

// Custom serialization for Activation
// We serialize it as a string identifier since function pointers can't be serialized
impl Serialize for Activation {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // For now, we only have SIGMOID
        // In the future, we could compare function pointers or use a registry
        serializer.serialize_str("sigmoid")
    }
}

impl<'de> Deserialize<'de> for Activation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        match s.as_str() {
            "sigmoid" => Ok(SIGMOID),
            _ => Err(serde::de::Error::custom(format!(
                "Unknown activation function: {}",
                s
            ))),
        }
    }
}
