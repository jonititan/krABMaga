//! A5 spatial index grid for agent-based modelling.
//!
//! This module provides sparse grid data structures backed by the
//! [A5 pentagonal spatial index](https://a5geo.org/). Agents are placed in
//! A5 cells (represented as `u64` indices) at a chosen resolution, and
//! neighbour queries exploit the A5 `grid_disk` operation.
//!
//! Enable with the **`a5grid`** Cargo feature:
//!
//! ```toml
//! krabmaga = { version = "0.5", features = ["a5grid"] }
//! ```
//!
//! # Example
//!
//! ```rust,ignore
//! use krabmaga::engine::fields::a5grid::SparseA5Grid;
//! use krabmaga::engine::fields::field::Field;
//!
//! let mut grid: SparseA5Grid<u32> = SparseA5Grid::new(3);
//! let cell = a5::lonlat_to_cell(a5::LonLat::new(-0.12, 51.5), 3).unwrap();
//! grid.set_object_location(42, cell);
//! grid.update();
//! assert_eq!(grid.get(cell), Some(42));
//! ```
//!
//! # Visualization support
//!
//! When using `SparseA5Grid` with the `visualization` or `visualization_wasm` features enabled,
//! implement the [`crate::visualization::agent_render::AgentRender`] trait to render agents.
//! Use the grid's coordinate conversion methods to map A5 cell indices to screen coordinates:
//!
//! ```rust,ignore
//! # use krabmaga::visualization::agent_render::AgentRender;
//! let cell = state.my_grid.get_location(&agent).unwrap();
//! let (lon, lat) = state.my_grid.cell_to_lonlat(cell);
//! // Convert geographic coordinates (lon, lat) to screen space based on simulation bounds
//! let screen_x = ((lon - min_lon) / (max_lon - min_lon)) * viewport_width;
//! let screen_y = ((lat - min_lat) / (max_lat - min_lat)) * viewport_height;
//! ```
//!
//! This allows for geographic visualization of agents distributed across a pentagonal spatial index.

use crate::engine::fields::field::Field;

use cfg_if::cfg_if;
use std::hash::Hash;

// ---------------------------------------------------------------------------
// Parallel / visualization variant  (DBDashMap-backed)
// ---------------------------------------------------------------------------
cfg_if! {
    if #[cfg(any(feature = "parallel", feature = "visualization", feature = "visualization_wasm"))] {
        use crate::utils::dbdashmap::DBDashMap;
        use crate::engine::fields::grid_option::GridOption;

        /// A sparse object grid indexed by A5 cell ids (`u64`), for parallel / visualization builds.
        ///
        /// Internally uses double-buffered concurrent maps identical to the ones
        /// used by [`SparseGrid2D`](super::sparse_object_grid_2d::SparseGrid2D).
        pub struct SparseA5Grid<O: Eq + Hash + Clone + Copy> {
            /// Map from object → cell id it occupies.
            pub obj2cell: DBDashMap<O, u64>,
            /// Map from cell id → objects at that cell.
            pub cell2objs: DBDashMap<u64, Vec<O>>,
            /// The A5 resolution of this grid (0–15).
            pub resolution: i32,
        }

        impl<O: Eq + Hash + Clone + Copy> SparseA5Grid<O> {

            /// Create a new, empty `SparseA5Grid` at the given resolution.
            ///
            /// Resolution must be a valid A5 resolution (0–30).
            pub fn new(resolution: i32) -> Self {
                SparseA5Grid {
                    obj2cell: DBDashMap::new(),
                    cell2objs: DBDashMap::new(),
                    resolution,
                }
            }

            /// Apply a closure to every `(cell, object)` pair.
            ///
            /// Signature matches the sequential variant so callers can write
            /// feature-portable code.
            pub fn apply_to_all_values<F>(&self, closure: F, option: GridOption)
            where
                F: Fn(&u64, &O),
            {
                if matches!(option, GridOption::READ | GridOption::READWRITE) {
                    let keys: Vec<u64> =
                        self.cell2objs.keys().into_iter().copied().collect();
                    for cell in keys {
                        if let Some(objs) = self.cell2objs.get_read(&cell) {
                            for obj in objs.iter() {
                                closure(&cell, obj);
                            }
                        }
                    }
                }
                if matches!(option, GridOption::WRITE | GridOption::READWRITE) {
                    let keys: Vec<u64> = self.cell2objs.w_keys();
                    for cell in keys {
                        if let Some(objs) = self.cell2objs.get_write(&cell) {
                            for obj in objs.iter() {
                                closure(&cell, obj);
                            }
                        }
                    }
                }
            }

            /// Return the first object found at `cell` (read state).
            pub fn get(&self, cell: u64) -> Option<O> {
                match self.cell2objs.get_read(&cell) {
                    Some(vec) => {
                        if vec.is_empty() { None } else { Some(vec[0]) }
                    }
                    None => None,
                }
            }

            /// Return all occupied cells in the read buffer.
            pub fn get_occupied_cells(&self) -> Vec<u64> {
                self.cell2objs.keys().into_iter().copied().collect()
            }

            /// Return all objects in the grid (read state).
            pub fn get_all_objects(&self) -> Vec<O> {
                let keys: Vec<u64> =
                    self.cell2objs.keys().into_iter().copied().collect();
                let mut all = Vec::new();
                for cell in keys {
                    if let Some(objs) = self.cell2objs.get_read(&cell) {
                        all.extend(objs.iter().cloned());
                    }
                }
                all
            }

            /// Number of distinct objects tracked in the read buffer.
            pub fn num_objects(&self) -> usize {
                self.obj2cell.r_len()
            }

            /// Number of distinct objects tracked in the write buffer.
            pub fn num_objects_unbuffered(&self) -> usize {
                self.obj2cell.len()
            }

            /// Return all objects at `cell` (read state).
            pub fn get_objects(&self, cell: u64) -> Option<Vec<O>> {
                match self.cell2objs.get_read(&cell) {
                    Some(vec) => {
                        if vec.is_empty() { None } else { Some(vec.clone()) }
                    }
                    None => None,
                }
            }

            /// Return all objects at `cell` (write state).
            pub fn get_objects_unbuffered(&self, cell: u64) -> Option<Vec<O>> {
                match self.cell2objs.get_write(&cell) {
                    Some(vec) => {
                        if vec.is_empty() { None } else { Some(vec.clone()) }
                    }
                    None => None,
                }
            }

            /// Get the cell that `obj` currently occupies (read state).
            pub fn get_location(&self, obj: &O) -> Option<u64> {
                self.obj2cell.get_read(obj).copied()
            }

            /// Get the cell that `obj` currently occupies (write state).
            pub fn get_location_unbuffered(&self, obj: &O) -> Option<u64> {
                self.obj2cell.get_write(obj).map(|r| *r)
            }

            /// Return the first object at `cell` (write state).
            pub fn get_unbuffered(&self, cell: u64) -> Option<O> {
                match self.cell2objs.get_write(&cell) {
                    Some(vec) => {
                        if vec.is_empty() { None } else { Some(vec[0]) }
                    }
                    None => None,
                }
            }

            /// Iterate over all `(object, cell)` pairs in the read buffer.
            pub fn iter_objects<F>(&self, closure: F)
            where
                F: Fn(&O, &u64),
            {
                let keys: Vec<O> =
                    self.obj2cell.keys().into_iter().copied().collect();
                for obj in keys {
                    if let Some(cell) = self.obj2cell.get_read(&obj) {
                        closure(&obj, cell);
                    }
                }
            }

            /// Iterate over all `(object, cell)` pairs in the write buffer.
            pub fn iter_objects_unbuffered<F>(&self, closure: F)
            where
                F: Fn(&O, &u64),
            {
                let keys: Vec<O> = self.obj2cell.w_keys();
                for obj in keys {
                    if let Some(cell) = self.obj2cell.get_write(&obj) {
                        closure(&obj, &*cell);
                    }
                }
            }

            /// Remove `obj` from the grid (write state).
            pub fn remove_object(&self, obj: &O) {
                if let Some(cell) = self.obj2cell.get_write(obj).map(|r| *r) {
                    self.obj2cell.remove(obj);
                    if let Some(mut vec) = self.cell2objs.get_write(&cell) {
                        vec.retain(|o| o != obj);
                    }
                }
            }

            /// Place `obj` at `new_cell`, removing it from any previous cell first.
            pub fn set_object_location(&self, obj: O, new_cell: u64) {
                // Remove from old cell if present.
                if let Some(old_cell) = self.obj2cell.get_write(&obj).map(|r| *r) {
                    if let Some(mut vec) = self.cell2objs.get_write(&old_cell) {
                        vec.retain(|o| *o != obj);
                    }
                }

                self.obj2cell.insert(obj, new_cell);
                match self.cell2objs.get_write(&new_cell) {
                    Some(mut vec) => {
                        vec.push(obj);
                    }
                    None => {
                        self.cell2objs.insert(new_cell, vec![obj]);
                    }
                }
            }

            // ----- A5-specific helpers -----

            /// Return the A5 grid-disk of `cell` at radius `k`, **including**
            /// `cell` itself. Thin wrapper over [`a5::grid_disk`].
            pub fn get_disk(&self, cell: u64, k: usize) -> Vec<u64> {
                a5::grid_disk(cell, k).unwrap_or_default()
            }

            /// Return all cells within topological distance `k` of `cell`,
            /// **excluding** `cell` itself. For `k == 0` this returns an empty
            /// vector. Matches the semantics of `SparseGrid2D::get_neighbors_*`.
            pub fn get_neighbors(&self, cell: u64, k: usize) -> Vec<u64> {
                self.get_disk(cell, k)
                    .into_iter()
                    .filter(|&c| c != cell)
                    .collect()
            }

            /// Return all objects in cells within topological distance `k` of
            /// `cell` (read state), **excluding** `cell` itself.
            pub fn get_neighbor_objects(&self, cell: u64, k: usize) -> Vec<O> {
                let mut result = Vec::new();
                for neighbor in self.get_neighbors(cell, k) {
                    if let Some(vec) = self.cell2objs.get_read(&neighbor) {
                        result.extend(vec);
                    }
                }
                result
            }

            /// Convert longitude/latitude (in degrees) to an A5 cell id at
            /// this grid's resolution.
            pub fn lonlat_to_cell(&self, lon: f64, lat: f64) -> u64 {
                a5::lonlat_to_cell(a5::LonLat::new(lon, lat), self.resolution)
                    .expect("lonlat_to_cell failed")
            }

            /// Convert an A5 cell id back to `(longitude, latitude)` in degrees.
            pub fn cell_to_lonlat(&self, cell: u64) -> (f64, f64) {
                let ll = a5::cell_to_lonlat(cell).expect("cell_to_lonlat failed");
                (ll.longitude.get(), ll.latitude.get())
            }

            /// Get the resolution stored in a cell index.
            pub fn cell_resolution(cell: u64) -> i32 {
                a5::get_resolution(cell)
            }
        }

        impl<O: Eq + Hash + Clone + Copy> Field for SparseA5Grid<O> {
            fn lazy_update(&mut self) {
                self.obj2cell.lazy_update();
                self.cell2objs.lazy_update();
            }

            fn update(&mut self) {
                self.obj2cell.update();
                self.cell2objs.update();
            }
        }

    } else {

// ---------------------------------------------------------------------------
// Sequential variant  (double-buffered HashMap)
// ---------------------------------------------------------------------------

        use std::cell::RefCell;
        use hashbrown::HashMap;
        use crate::engine::fields::grid_option::GridOption;

        /// A sparse object grid indexed by A5 cell ids (`u64`), for sequential (single-threaded) builds.
        ///
        /// Uses a double-buffered pair of `HashMap`s, identical to the pattern
        /// used by [`SparseGrid2D`](super::sparse_object_grid_2d::SparseGrid2D).
        pub struct SparseA5Grid<O: Eq + Hash + Clone + Copy> {
            /// Object → cell maps (double-buffered).
            pub obj2cell: Vec<RefCell<HashMap<O, u64>>>,
            /// Cell → objects maps (double-buffered).
            pub cell2objs: Vec<RefCell<HashMap<u64, Vec<O>>>>,
            read: usize,
            write: usize,
            /// The A5 resolution of this grid (0–15).
            pub resolution: i32,
        }

        impl<O: Eq + Hash + Clone + Copy> SparseA5Grid<O> {

            /// Create a new, empty `SparseA5Grid` at the given resolution.
            ///
            /// Resolution must be a valid A5 resolution (0–30).
            pub fn new(resolution: i32) -> Self {
                SparseA5Grid {
                    obj2cell: vec![RefCell::new(HashMap::new()), RefCell::new(HashMap::new())],
                    cell2objs: vec![RefCell::new(HashMap::new()), RefCell::new(HashMap::new())],
                    read: 0,
                    write: 1,
                    resolution,
                }
            }

            /// Apply a closure to every (cell, object) pair.
            pub fn apply_to_all_values<F>(&self, closure: F, option: GridOption)
            where
                F: Fn(&u64, &O),
            {
                match option {
                    GridOption::READ => {
                        let cell2objs = self.cell2objs[self.read].borrow();
                        for (cell, objs) in cell2objs.iter() {
                            for obj in objs {
                                closure(cell, obj);
                            }
                        }
                    }
                    GridOption::WRITE => {
                        let cell2objs = self.cell2objs[self.write].borrow();
                        for (cell, objs) in cell2objs.iter() {
                            for obj in objs {
                                closure(cell, obj);
                            }
                        }
                    }
                    GridOption::READWRITE => {
                        let r = self.cell2objs[self.read].borrow();
                        for (cell, objs) in r.iter() {
                            for obj in objs {
                                closure(cell, obj);
                            }
                        }
                        let w = self.cell2objs[self.write].borrow();
                        for (cell, objs) in w.iter() {
                            for obj in objs {
                                closure(cell, obj);
                            }
                        }
                    }
                }
            }

            /// Return the first object at `cell` (read state).
            pub fn get(&self, cell: u64) -> Option<O> {
                let cell2objs = self.cell2objs[self.read].borrow();
                match cell2objs.get(&cell) {
                    Some(vec) if !vec.is_empty() => Some(vec[0]),
                    _ => None,
                }
            }

            /// Return all objects at `cell` (read state).
            pub fn get_objects(&self, cell: u64) -> Option<Vec<O>> {
                let cell2objs = self.cell2objs[self.read].borrow();
                match cell2objs.get(&cell) {
                    Some(vec) if !vec.is_empty() => Some(vec.to_vec()),
                    _ => None,
                }
            }

            /// Return all objects at `cell` (write state).
            pub fn get_objects_unbuffered(&self, cell: u64) -> Option<Vec<O>> {
                let cell2objs = self.cell2objs[self.write].borrow();
                match cell2objs.get(&cell) {
                    Some(vec) if !vec.is_empty() => Some(vec.to_vec()),
                    _ => None,
                }
            }

            /// Get the cell that `obj` currently occupies (read state).
            pub fn get_location(&self, obj: &O) -> Option<u64> {
                let obj2cell = self.obj2cell[self.read].borrow();
                obj2cell.get(obj).copied()
            }

            /// Get the cell that `obj` currently occupies (write state).
            pub fn get_location_unbuffered(&self, obj: &O) -> Option<u64> {
                let obj2cell = self.obj2cell[self.write].borrow();
                obj2cell.get(obj).copied()
            }

            /// Return the first object at `cell` (write state).
            pub fn get_unbuffered(&self, cell: u64) -> Option<O> {
                let cell2objs = self.cell2objs[self.write].borrow();
                match cell2objs.get(&cell) {
                    Some(vec) if !vec.is_empty() => Some(vec[0]),
                    _ => None,
                }
            }

            /// Iterate over all (object, cell) pairs (read state).
            pub fn iter_objects<F>(&self, closure: F)
            where
                F: Fn(&O, &u64),
            {
                let obj2cell = self.obj2cell[self.read].borrow();
                for (obj, cell) in obj2cell.iter() {
                    closure(obj, cell);
                }
            }

            /// Iterate over all (object, cell) pairs (write state).
            pub fn iter_objects_unbuffered<F>(&self, closure: F)
            where
                F: Fn(&O, &u64),
            {
                let obj2cell = self.obj2cell[self.write].borrow();
                for (obj, cell) in obj2cell.iter() {
                    closure(obj, cell);
                }
            }

            /// Remove `obj` from the grid (write state).
            pub fn remove_object(&self, obj: &O) {
                let mut obj2cell = self.obj2cell[self.write].borrow_mut();
                let mut cell2objs = self.cell2objs[self.write].borrow_mut();
                if let Some(cell) = obj2cell.remove(obj) {
                    if let Some(vec) = cell2objs.get_mut(&cell) {
                        vec.retain(|o| o != obj);
                    }
                }
            }

            /// Place `obj` at `new_cell`, removing it from any previous cell first (write state).
            pub fn set_object_location(&self, obj: O, new_cell: u64) {
                let mut obj2cell = self.obj2cell[self.write].borrow_mut();
                let mut cell2objs = self.cell2objs[self.write].borrow_mut();

                // Remove from old cell if present.
                if let Some(old_cell) = obj2cell.remove(&obj) {
                    if let Some(vec) = cell2objs.get_mut(&old_cell) {
                        vec.retain(|o| *o != obj);
                    }
                }

                obj2cell.insert(obj, new_cell);
                cell2objs
                    .entry(new_cell)
                    .or_insert_with(Vec::new)
                    .push(obj);
            }

            /// Return the number of objects in the grid (read state).
            pub fn num_objects(&self) -> usize {
                self.obj2cell[self.read].borrow().len()
            }

            /// Return the number of objects in the grid (write state).
            pub fn num_objects_unbuffered(&self) -> usize {
                self.obj2cell[self.write].borrow().len()
            }

            /// Return all cells that have at least one object (read state).
            pub fn get_occupied_cells(&self) -> Vec<u64> {
                let cell2objs = self.cell2objs[self.read].borrow();
                cell2objs.keys().cloned().collect()
            }

            /// Return all objects as a flat vector (read state).
            pub fn get_all_objects(&self) -> Vec<O> {
                let obj2cell = self.obj2cell[self.read].borrow();
                obj2cell.keys().cloned().collect()
            }

            // ----- A5-specific helpers -----

            /// Return the A5 grid-disk of `cell` at radius `k`, **including**
            /// `cell` itself. Thin wrapper over [`a5::grid_disk`].
            pub fn get_disk(&self, cell: u64, k: usize) -> Vec<u64> {
                a5::grid_disk(cell, k).unwrap_or_default()
            }

            /// Return all cells within topological distance `k` of `cell`,
            /// **excluding** `cell` itself. For `k == 0` this returns an empty
            /// vector. Matches the semantics of `SparseGrid2D::get_neighbors_*`.
            pub fn get_neighbors(&self, cell: u64, k: usize) -> Vec<u64> {
                self.get_disk(cell, k)
                    .into_iter()
                    .filter(|&c| c != cell)
                    .collect()
            }

            /// Return all objects in cells within topological distance `k` of
            /// `cell` (read state), **excluding** `cell` itself.
            pub fn get_neighbor_objects(&self, cell: u64, k: usize) -> Vec<O> {
                let cell2objs = self.cell2objs[self.read].borrow();
                let mut result = Vec::new();
                for neighbor in self.get_neighbors(cell, k) {
                    if let Some(vec) = cell2objs.get(&neighbor) {
                        result.extend(vec);
                    }
                }
                result
            }

            /// Convert longitude/latitude to an A5 cell id at this grid's resolution.
            ///
            /// `lon` is longitude in degrees, `lat` is latitude in degrees.
            pub fn lonlat_to_cell(&self, lon: f64, lat: f64) -> u64 {
                a5::lonlat_to_cell(a5::LonLat::new(lon, lat), self.resolution)
                    .expect("lonlat_to_cell failed")
            }

            /// Convert an A5 cell id back to (longitude, latitude) in degrees.
            pub fn cell_to_lonlat(&self, cell: u64) -> (f64, f64) {
                let ll = a5::cell_to_lonlat(cell).expect("cell_to_lonlat failed");
                (ll.longitude.get(), ll.latitude.get())
            }

            /// Get the resolution stored in a cell index.
            pub fn cell_resolution(cell: u64) -> i32 {
                a5::get_resolution(cell)
            }
        }

        impl<O: Eq + Hash + Clone + Copy> Field for SparseA5Grid<O> {
            /// Swap read/write buffers and clear the new write buffer.
            /// Matches `SparseGrid2D::lazy_update`.
            fn lazy_update(&mut self) {
                std::mem::swap(&mut self.read, &mut self.write);
                self.obj2cell[self.write].borrow_mut().clear();
                self.cell2objs[self.write].borrow_mut().clear();
            }

            /// Eagerly copy the write buffer into the read buffer and clear
            /// the write buffer. Matches `SparseGrid2D::update` semantics:
            /// agents must re-register their location each step, otherwise
            /// they disappear from the grid after the next update.
            fn update(&mut self) {
                {
                    let mut r_obj = self.obj2cell[self.read].borrow_mut();
                    r_obj.clear();
                    for (k, v) in self.obj2cell[self.write].borrow().iter() {
                        r_obj.insert(*k, *v);
                    }
                }
                {
                    let mut r_cell = self.cell2objs[self.read].borrow_mut();
                    r_cell.clear();
                    for (k, v) in self.cell2objs[self.write].borrow().iter() {
                        r_cell.insert(*k, v.clone());
                    }
                }
                self.obj2cell[self.write].borrow_mut().clear();
                self.cell2objs[self.write].borrow_mut().clear();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<O: Eq + Hash + Clone + Copy + std::fmt::Display> std::fmt::Display for SparseA5Grid<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SparseA5Grid<resolution={}>", self.resolution)
    }
}

// ---------------------------------------------------------------------------
// Unit tests (sequential only — the parallel tests would need threading setup)
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(not(any(
    feature = "parallel",
    feature = "visualization",
    feature = "visualization_wasm"
)))]
mod tests {
    use super::*;

    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    struct TestObj {
        id: i32,
    }

    /// Helper: produce a cell id by indexing a known location at resolution 3.
    fn sample_cell(lon: f64, lat: f64) -> u64 {
        a5::lonlat_to_cell(a5::LonLat::new(lon, lat), 3).unwrap()
    }

    // -- basic set / get ---------------------------------------------------

    #[test]
    fn test_set_and_get() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let obj = TestObj { id: 1 };
        let cell = sample_cell(0.0, 51.5);

        grid.set_object_location(obj, cell);
        grid.update();

        assert_eq!(grid.get(cell), Some(obj));
        assert_eq!(grid.get_location(&obj), Some(cell));
    }

    #[test]
    fn test_get_empty_cell() {
        let grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let cell = sample_cell(0.0, 51.5);
        assert_eq!(grid.get(cell), None);
    }

    // -- remove ------------------------------------------------------------

    #[test]
    fn test_remove() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let obj = TestObj { id: 1 };
        let cell = sample_cell(10.0, 48.0);

        grid.set_object_location(obj, cell);
        grid.update();

        grid.remove_object(&obj);
        grid.update();

        assert_eq!(grid.get(cell), None);
        assert_eq!(grid.get_location(&obj), None);
    }

    // -- double-buffer semantics -------------------------------------------

    #[test]
    fn test_lazy_update() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let obj = TestObj { id: 1 };
        let cell = sample_cell(0.0, 0.0);

        grid.set_object_location(obj, cell);
        grid.lazy_update();

        // After lazy_update the write buffer became read, so object is visible.
        assert_eq!(grid.get(cell), Some(obj));
        assert_eq!(grid.get_location(&obj), Some(cell));

        // Another lazy_update clears the (now-write) buffer, and the old
        // write (now read) was already empty => nothing visible.
        grid.lazy_update();
        assert_eq!(grid.get(cell), None);
    }

    #[test]
    fn test_update_commits_and_clears_write() {
        // Matches SparseGrid2D semantics: `update` copies write→read then
        // clears write, so a second consecutive `update` with no new writes
        // drains the grid entirely. Agents must re-register each step.
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let obj = TestObj { id: 1 };
        let cell = sample_cell(0.0, 0.0);

        grid.set_object_location(obj, cell);
        grid.update();
        assert_eq!(grid.get(cell), Some(obj));
        assert_eq!(grid.get_unbuffered(cell), None);

        grid.update();
        assert_eq!(grid.get(cell), None);
    }

    // -- multiple objects in one cell --------------------------------------

    #[test]
    fn test_multiple_objects_same_cell() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let obj1 = TestObj { id: 1 };
        let obj2 = TestObj { id: 2 };
        let cell = sample_cell(0.0, 0.0);

        grid.set_object_location(obj1, cell);
        grid.set_object_location(obj2, cell);
        grid.update();

        let objects = grid.get_objects(cell).unwrap();
        assert_eq!(objects.len(), 2);
        assert!(objects.contains(&obj1));
        assert!(objects.contains(&obj2));
    }

    // -- move object -------------------------------------------------------

    #[test]
    fn test_move_object() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let obj = TestObj { id: 1 };
        let cell1 = sample_cell(0.0, 0.0);
        let cell2 = sample_cell(10.0, 10.0);

        grid.set_object_location(obj, cell1);
        grid.update();

        grid.set_object_location(obj, cell2);
        grid.update();

        assert_eq!(grid.get_location(&obj), Some(cell2));
        assert_eq!(grid.get(cell1), None);
        assert_eq!(grid.get(cell2), Some(obj));
    }

    // -- A5 neighbour queries ----------------------------------------------

    #[test]
    fn test_disk_and_neighbors_center_semantics() {
        // `a5::grid_disk` only includes the center at k == 0. For k >= 1 the
        // returned set already excludes it. `get_neighbors` enforces that
        // center-free contract uniformly, so it is empty at k == 0 and
        // otherwise identical to `get_disk`.
        let grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let cell = grid.lonlat_to_cell(-0.12, 51.5);

        let disk0 = grid.get_disk(cell, 0);
        assert_eq!(disk0, vec![cell]);
        assert!(grid.get_neighbors(cell, 0).is_empty());

        let disk1 = grid.get_disk(cell, 1);
        let neigh1 = grid.get_neighbors(cell, 1);
        assert!(!disk1.is_empty());
        assert!(!disk1.contains(&cell));
        assert_eq!(disk1, neigh1);
        assert!(!neigh1.contains(&cell));
    }

    #[test]
    fn test_neighbor_objects() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let center = grid.lonlat_to_cell(-0.12, 51.5);
        let neighbors = grid.get_neighbors(center, 1);

        let mut placed = Vec::new();
        for (i, &n) in neighbors.iter().enumerate() {
            let obj = TestObj { id: i as i32 };
            grid.set_object_location(obj, n);
            placed.push(obj);
        }
        grid.update();

        let found = grid.get_neighbor_objects(center, 1);
        assert_eq!(found.len(), placed.len());
    }

    // -- coordinate conversion helpers -------------------------------------

    #[test]
    fn test_lonlat_roundtrip() {
        let grid: SparseA5Grid<TestObj> = SparseA5Grid::new(5);
        let cell = grid.lonlat_to_cell(-0.12, 51.5);
        let (lon, lat) = grid.cell_to_lonlat(cell);
        // Round-trip should be close (within cell size at res 5)
        assert!((lon - (-0.12)).abs() < 1.0, "lon delta too large: {}", lon);
        assert!((lat - 51.5).abs() < 1.0, "lat delta too large: {}", lat);
    }

    #[test]
    fn test_cell_resolution() {
        let grid: SparseA5Grid<TestObj> = SparseA5Grid::new(7);
        let cell = grid.lonlat_to_cell(0.0, 0.0);
        assert_eq!(SparseA5Grid::<TestObj>::cell_resolution(cell), 7);
    }

    // -- num_objects / get helpers -----------------------------------------

    #[test]
    fn test_num_objects() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        assert_eq!(grid.num_objects(), 0);

        let obj = TestObj { id: 1 };
        let cell = sample_cell(0.0, 0.0);
        grid.set_object_location(obj, cell);
        grid.update();

        assert_eq!(grid.num_objects(), 1);
    }

    #[test]
    fn test_get_occupied_cells() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let cell1 = sample_cell(0.0, 0.0);
        let cell2 = sample_cell(10.0, 10.0);

        grid.set_object_location(TestObj { id: 1 }, cell1);
        grid.set_object_location(TestObj { id: 2 }, cell2);
        grid.update();

        let occupied = grid.get_occupied_cells();
        assert_eq!(occupied.len(), 2);
        assert!(occupied.contains(&cell1));
        assert!(occupied.contains(&cell2));
    }

    #[test]
    fn test_get_all_objects() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let obj1 = TestObj { id: 1 };
        let obj2 = TestObj { id: 2 };

        grid.set_object_location(obj1, sample_cell(0.0, 0.0));
        grid.set_object_location(obj2, sample_cell(10.0, 10.0));
        grid.update();

        let all = grid.get_all_objects();
        assert_eq!(all.len(), 2);
        assert!(all.contains(&obj1));
        assert!(all.contains(&obj2));
    }

    #[test]
    fn test_iter_objects() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let obj1 = TestObj { id: 1 };
        let obj2 = TestObj { id: 2 };
        let cell1 = sample_cell(0.0, 0.0);
        let cell2 = sample_cell(10.0, 10.0);

        grid.set_object_location(obj1, cell1);
        grid.set_object_location(obj2, cell2);
        grid.update();

        let count = std::cell::Cell::new(0usize);
        grid.iter_objects(|_obj, _cell| {
            count.set(count.get() + 1);
        });
        assert_eq!(count.get(), 2);
    }

    #[test]
    fn test_apply_to_all_values() {
        let mut grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let obj1 = TestObj { id: 1 };
        let obj2 = TestObj { id: 2 };
        let cell = sample_cell(0.0, 0.0);

        grid.set_object_location(obj1, cell);
        grid.set_object_location(obj2, cell);
        grid.update();

        let count = std::cell::Cell::new(0usize);
        grid.apply_to_all_values(|_cell, _obj| {
            count.set(count.get() + 1);
        }, GridOption::READ);
        assert_eq!(count.get(), 2);
    }

    #[test]
    fn test_display() {
        let grid: SparseA5Grid<i32> = SparseA5Grid::new(5);
        let s = format!("{}", grid);
        assert!(s.contains("SparseA5Grid"));
        assert!(s.contains("5"));
    }

    // -- unbuffered accessors ----------------------------------------------

    #[test]
    fn test_unbuffered_accessors() {
        let grid: SparseA5Grid<TestObj> = SparseA5Grid::new(3);
        let obj = TestObj { id: 42 };
        let cell = sample_cell(5.0, 45.0);

        // Write-side should see the object before update
        grid.set_object_location(obj, cell);
        assert_eq!(grid.get_unbuffered(cell), Some(obj));
        assert_eq!(grid.get_location_unbuffered(&obj), Some(cell));
        assert_eq!(grid.get_objects_unbuffered(cell).unwrap().len(), 1);
        assert_eq!(grid.num_objects_unbuffered(), 1);

        // Read-side should NOT see the object before update
        assert_eq!(grid.get(cell), None);
        assert_eq!(grid.num_objects(), 0);
    }
}