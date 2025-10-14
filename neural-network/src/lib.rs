#[macro_use]
extern crate derive_builder;
pub mod network;
pub mod activations;
pub mod examples;
pub mod checkpoint;

pub mod matrix {

    pub use matrix::matrix::Matrix;
}