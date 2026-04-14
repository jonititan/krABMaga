#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel"
)))]
use krabmaga::{
    engine::fields::a5grid::SparseA5Grid, engine::fields::field::Field,
    engine::fields::grid_option::GridOption,
};

#[cfg(feature = "a5grid")]
#[derive(Clone, Copy, PartialEq, Eq, std::hash::Hash, Debug)]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel"
)))]
struct TestAgent {
    id: u32,
}

/// Helper to convert lon/lat to A5 cell at resolution 3
#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
fn sample_cell(lon: f64, lat: f64) -> u64 {
    a5::lonlat_to_cell(a5::LonLat::new(lon, lat), 3).unwrap()
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_set_and_get() {
    let mut grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let agent = TestAgent { id: 1 };
    let cell = sample_cell(0.0, 51.5);

    grid.set_object_location(agent, cell);
    grid.update();

    assert_eq!(grid.get(cell), Some(agent));
    assert_eq!(grid.get_location(&agent), Some(cell));
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_get_empty_cell() {
    let grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let cell = sample_cell(0.0, 51.5);
    assert_eq!(grid.get(cell), None);
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_remove_object() {
    let mut grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let agent = TestAgent { id: 1 };
    let cell = sample_cell(10.0, 48.0);

    grid.set_object_location(agent, cell);
    grid.update();

    grid.remove_object(&agent);
    grid.update();

    assert_eq!(grid.get(cell), None);
    assert_eq!(grid.get_location(&agent), None);
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_multiple_objects_same_cell() {
    let mut grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let agent1 = TestAgent { id: 1 };
    let agent2 = TestAgent { id: 2 };
    let cell = sample_cell(0.0, 0.0);

    grid.set_object_location(agent1, cell);
    grid.set_object_location(agent2, cell);
    grid.update();

    let objects = grid.get_objects(cell).unwrap();
    assert_eq!(objects.len(), 2);
    assert!(objects.contains(&agent1));
    assert!(objects.contains(&agent2));
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_move_object() {
    let mut grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let agent = TestAgent { id: 1 };
    let cell1 = sample_cell(0.0, 0.0);
    let cell2 = sample_cell(10.0, 10.0);

    grid.set_object_location(agent, cell1);
    grid.update();

    grid.set_object_location(agent, cell2);
    grid.update();

    assert_eq!(grid.get_location(&agent), Some(cell2));
    assert_eq!(grid.get(cell1), None);
    assert_eq!(grid.get(cell2), Some(agent));
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_get_all_objects() {
    let mut grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let agent1 = TestAgent { id: 1 };
    let agent2 = TestAgent { id: 2 };

    grid.set_object_location(agent1, sample_cell(0.0, 0.0));
    grid.set_object_location(agent2, sample_cell(10.0, 10.0));
    grid.update();

    let all = grid.get_all_objects();
    assert_eq!(all.len(), 2);
    assert!(all.contains(&agent1));
    assert!(all.contains(&agent2));
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_iter_objects() {
    let mut grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let agent1 = TestAgent { id: 1 };
    let agent2 = TestAgent { id: 2 };
    let cell1 = sample_cell(0.0, 0.0);
    let cell2 = sample_cell(10.0, 10.0);

    grid.set_object_location(agent1, cell1);
    grid.set_object_location(agent2, cell2);
    grid.update();

    let count = std::cell::Cell::new(0usize);
    grid.iter_objects(|_obj, _cell| {
        count.set(count.get() + 1);
    });
    assert_eq!(count.get(), 2);
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_apply_to_all_values() {
    let mut grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let agent1 = TestAgent { id: 1 };
    let agent2 = TestAgent { id: 2 };
    let cell = sample_cell(0.0, 0.0);

    grid.set_object_location(agent1, cell);
    grid.set_object_location(agent2, cell);
    grid.update();

    let count = std::cell::Cell::new(0usize);
    grid.apply_to_all_values(
        |_cell, _obj| {
            count.set(count.get() + 1);
        },
        GridOption::READ,
    );
    assert_eq!(count.get(), 2);
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_get_neighbors() {
    let grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let cell = grid.lonlat_to_cell(-0.12, 51.5);
    let neighbors = grid.get_neighbors(cell, 1);

    // grid_disk(1) returns nearby cells
    assert!(!neighbors.is_empty());
    assert!(neighbors.len() > 1);
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_get_neighbor_objects() {
    let mut grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let center = grid.lonlat_to_cell(-0.12, 51.5);
    let disk = grid.get_neighbors(center, 1);

    // Place objects in neighbour cells (not the center)
    let mut placed = Vec::new();
    for (i, &n) in disk.iter().enumerate() {
        if n != center {
            let agent = TestAgent { id: i as u32 };
            grid.set_object_location(agent, n);
            placed.push(agent);
        }
    }
    grid.update();

    let found = grid.get_neighbor_objects(center, 1);
    assert_eq!(found.len(), placed.len());
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_lonlat_roundtrip() {
    let grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(5);
    let lon = -0.12;
    let lat = 51.5;

    // Test that conversion works without panicking
    let cell = grid.lonlat_to_cell(lon, lat);
    let (lon2, lat2) = grid.cell_to_lonlat(cell);

    // A5 grid cells have geographic extent, so roundtrip conversion
    // doesn't guarantee exact coordinates - just that we get a valid result
    assert!(lon2.is_finite());
    assert!(lat2.is_finite());
    assert!(lon2 >= -180.0 && lon2 <= 180.0);
    assert!(lat2 >= -90.0 && lat2 <= 90.0);
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_lazy_update_semantics() {
    let mut grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let agent = TestAgent { id: 1 };
    let cell = sample_cell(0.0, 0.0);

    grid.set_object_location(agent, cell);
    grid.lazy_update();

    // After lazy_update the write buffer became read, so object is visible.
    assert_eq!(grid.get(cell), Some(agent));
    assert_eq!(grid.get_location(&agent), Some(cell));

    // Another lazy_update clears the (now-write) buffer, and the old
    // write (now read) was already empty => nothing visible.
    grid.lazy_update();
    assert_eq!(grid.get(cell), None);
}

#[cfg(feature = "a5grid")]
#[cfg(not(any(
    feature = "visualization",
    feature = "visualization_wasm",
    feature = "parallel",
)))]
#[test]
fn a5grid_update_commits_and_clears_write() {
    // Matches SparseGrid2D semantics: `update` copies write→read and then
    // clears write. A second consecutive `update` with no new writes drains
    // the grid. Agents must re-register their location every step.
    let mut grid: SparseA5Grid<TestAgent> = SparseA5Grid::new(3);
    let agent = TestAgent { id: 1 };
    let cell = sample_cell(0.0, 0.0);

    grid.set_object_location(agent, cell);
    grid.update();
    assert_eq!(grid.get(cell), Some(agent));

    grid.update();
    assert_eq!(grid.get(cell), None);
}