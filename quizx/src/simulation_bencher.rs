use std::cmp::min;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::time::Instant;

use crate::cli::CliError;
use crate::decompose::{
    BssTOnlyDriver, BssWithCatsDriver, Decomposer, Driver, DynamicTDriver, SimpFunc,
};
use crate::generate;
use crate::graph::{BasisElem, GraphLike};
use crate::simplify;
use crate::vec_graph::Graph as VecGraph;

// For plotting
// use plotters::prelude::*;
use std::collections::HashMap;
// use std::path::Path;

const SEED: u64 = 42;
const MIN_TCOUNT: usize = 6;
const MAX_TCOUNT: usize = 40;
const SAMPLES_PER_TCOUNT: usize = 4;

fn get_testset() -> Vec<VecGraph> {
    let mut graph_bins: [Vec<VecGraph>; MAX_TCOUNT - MIN_TCOUNT] =
        [(); MAX_TCOUNT - MIN_TCOUNT].map(|_| Vec::new());
    let mut count_full = 0;

    let mut circuit_builder = generate::RandomCircuitBuilder {
        ..Default::default()
    };
    circuit_builder.seed(SEED).qubits(10);
    circuit_builder.clifford_t(0.3);
    while count_full < MAX_TCOUNT - MIN_TCOUNT {
        for i in MIN_TCOUNT..10 * MAX_TCOUNT {
            circuit_builder.depth(i);
            let mut graph: VecGraph = circuit_builder.build().to_graph();
            graph.plug_inputs(&[BasisElem::Z0; 10]);
            graph.plug_outputs(&[BasisElem::Z0; 10]);
            simplify::full_simp(&mut graph);
            let t_count = graph.tcount();
            if (MIN_TCOUNT..MAX_TCOUNT).contains(&t_count)
                && graph_bins[t_count - MIN_TCOUNT].len() < SAMPLES_PER_TCOUNT
            {
                graph_bins[t_count - MIN_TCOUNT].push(graph);
                if graph_bins[t_count - MIN_TCOUNT].len() == SAMPLES_PER_TCOUNT {
                    count_full += 1;
                    println!("Full: {}", t_count)
                }
            }
        }
    }
    println!("Full!");
    graph_bins.into_iter().flatten().collect()
}

fn bench_setup(
    name: &str,
    mut max_tcount: usize,
    driver: &impl Driver,
    simpfunc: SimpFunc,
    testset: &[VecGraph],
) {
    // Prepare CSV files for this benchmark
    let filename_nterms = format!("benches/results/benchmark_alpha_{}.csv", name);
    let filename_times = format!("benches/results/benchmark_times_{}.csv", name);

    let mut file_nterms =
        BufWriter::new(File::create(&filename_nterms).expect("Could not create nterms CSV file"));
    let mut file_times =
        BufWriter::new(File::create(&filename_times).expect("Could not create times CSV file"));

    writeln!(file_nterms, "t_count,nterms").unwrap();
    writeln!(file_times, "t_count,runtime_nanos").unwrap();

    max_tcount = min(max_tcount, MAX_TCOUNT);

    for t_count in MIN_TCOUNT..max_tcount {
        let index = t_count - MIN_TCOUNT;
        println!("Benchmarking {} with t_count={}", name, t_count);

        for j in 0..SAMPLES_PER_TCOUNT {
            let graph = &testset[index * SAMPLES_PER_TCOUNT + j];

            // Time the decomposition
            let start = Instant::now();
            let mut decomposer = Decomposer::new(graph);
            decomposer
                .with_simp(simpfunc)
                .with_split_graphs_components(true)
                .decompose(driver);
            let elapsed = start.elapsed();

            // Save both nterms and runtime
            writeln!(file_nterms, "{},{}", t_count, decomposer.nterms).unwrap();
            writeln!(file_times, "{},{}", t_count, elapsed.as_nanos()).unwrap();
        }
    }
}

fn create_svg_plot(
    plot_type: &str,
    files: Vec<String>,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Load data from CSV files
    let mut data_series: HashMap<String, Vec<(f64, f64)>> = HashMap::new();

    for file in files {
        let name = file
            .split('/')
            .next_back()
            .unwrap()
            .replace("benchmark_", "")
            .replace(&format!("{}_", plot_type), "")
            .replace(".csv", "");

        let mut reader = csv::Reader::from_path(&file)?;
        let mut data_points: HashMap<usize, Vec<f64>> = HashMap::new();

        for result in reader.records() {
            let record = result?;
            let t_count: usize = record[0].parse()?;
            let value: f64 = record[1].parse()?;

            data_points.entry(t_count).or_default().push(value);
        }

        // Calculate mean for each t_count
        let mut series_data: Vec<(f64, f64)> = Vec::new();
        for (t_count, values) in data_points {
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let plot_value = if plot_type == "alpha" {
                mean.ln() // log of nterms
            } else {
                (mean / 1_000_000.0).ln() // log of runtime in milliseconds
            };
            series_data.push((t_count as f64, plot_value));
        }
        series_data.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        data_series.insert(name, series_data);
    }

    // SVG dimensions and margins
    let width = 800.0;
    let height = 600.0;
    let margin = 50.0;
    let plot_width = width - 2.0 * margin;
    let plot_height = height - 2.0 * margin;

    // Find data range
    let x_min = MIN_TCOUNT as f64;
    let x_max = MAX_TCOUNT as f64 - 1.0;
    let y_min = data_series
        .values()
        .flat_map(|v| v.iter().map(|(_, y)| *y))
        .fold(f64::INFINITY, f64::min);
    let y_max = data_series
        .values()
        .flat_map(|v| v.iter().map(|(_, y)| *y))
        .fold(f64::NEG_INFINITY, f64::max);

    // Create SVG
    let mut svg = String::new();
    svg.push_str(&format!(
        r#"<svg width="{}" height="{}" xmlns="http://www.w3.org/2000/svg">"#,
        width, height
    ));

    // White background
    svg.push_str(&format!(
        r#"<rect width="{}" height="{}" fill="white"/>"#,
        width, height
    ));

    // Title
    let title = if plot_type == "alpha" {
        "Average log(n_terms) vs t_count"
    } else {
        "Average log(runtime) vs t_count"
    };
    svg.push_str(&format!(r#"<text x="{}" y="30" text-anchor="middle" font-size="20" font-family="sans-serif">{}</text>"#, width/2.0, title));

    // Draw axes
    svg.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="2"/>"#,
        margin,
        margin,
        margin,
        height - margin
    ));
    svg.push_str(&format!(
        r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="black" stroke-width="2"/>"#,
        margin,
        height - margin,
        width - margin,
        height - margin
    ));

    // Y-axis label
    let y_label = if plot_type == "alpha" {
        "log(mean n_terms)"
    } else {
        "log(runtime in ms)"
    };
    svg.push_str(&format!(r#"<text x="20" y="{}" text-anchor="middle" font-size="14" font-family="sans-serif" transform="rotate(-90 20 {})">{}</text>"#, 
        height/2.0, height/2.0, y_label));

    // X-axis label
    svg.push_str(&format!(r#"<text x="{}" y="{}" text-anchor="middle" font-size="14" font-family="sans-serif">t_count</text>"#, 
        width/2.0, height - 10.0));

    // Grid lines and labels
    for i in 0..=5 {
        let x = margin + (i as f64 / 5.0) * plot_width;
        let x_val = x_min + (i as f64 / 5.0) * (x_max - x_min);
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="lightgray" stroke-width="1"/>"#,
            x,
            margin,
            x,
            height - margin
        ));
        svg.push_str(&format!(r#"<text x="{}" y="{}" text-anchor="middle" font-size="12" font-family="sans-serif">{:.0}</text>"#, 
            x, height - margin + 20.0, x_val));
    }

    for i in 0..=5 {
        let y = margin + (i as f64 / 5.0) * plot_height;
        let y_val = y_max - (i as f64 / 5.0) * (y_max - y_min);
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="lightgray" stroke-width="1"/>"#,
            margin,
            y,
            width - margin,
            y
        ));
        svg.push_str(&format!(r#"<text x="{}" y="{}" text-anchor="end" font-size="12" font-family="sans-serif">{:.2}</text>"#, 
            margin - 10.0, y + 5.0, y_val));
    }

    // Plot data
    let colors = ["red", "blue", "green", "magenta", "cyan"];
    let mut legend_y = 60.0;

    for (idx, (name, data)) in data_series.iter().enumerate() {
        let color = colors[idx % colors.len()];

        // Draw line
        let mut path = String::from("M");
        for (i, &(x_val, y_val)) in data.iter().enumerate() {
            let x = margin + ((x_val - x_min) / (x_max - x_min)) * plot_width;
            let y = margin + ((y_max - y_val) / (y_max - y_min)) * plot_height;

            if i == 0 {
                path.push_str(&format!(" {} {}", x, y));
            } else {
                path.push_str(&format!(" L {} {}", x, y));
            }
        }
        svg.push_str(&format!(
            r#"<path d="{}" fill="none" stroke="{}" stroke-width="2"/>"#,
            path, color
        ));

        // Draw points
        for &(x_val, y_val) in data {
            let x = margin + ((x_val - x_min) / (x_max - x_min)) * plot_width;
            let y = margin + ((y_max - y_val) / (y_max - y_min)) * plot_height;
            svg.push_str(&format!(
                r#"<circle cx="{}" cy="{}" r="3" fill="{}"/>"#,
                x, y, color
            ));
        }

        // Legend
        svg.push_str(&format!(
            r#"<line x1="{}" y1="{}" x2="{}" y2="{}" stroke="{}" stroke-width="2"/>"#,
            width - margin - 100.0,
            legend_y,
            width - margin - 80.0,
            legend_y,
            color
        ));
        svg.push_str(&format!(
            r#"<text x="{}" y="{}" font-size="12" font-family="sans-serif">{}</text>"#,
            width - margin - 75.0,
            legend_y + 4.0,
            name
        ));
        legend_y += 20.0;
    }

    svg.push_str("</svg>");

    // Write to file
    let mut file = File::create(output_path)?;
    file.write_all(svg.as_bytes())?;

    Ok(())
}

fn benchmark_driver(testset: &[VecGraph]) {
    println!("Running Driver benchmarks...");

    bench_setup(
        "BssTOnly",
        32,
        &BssTOnlyDriver { random_t: false },
        SimpFunc::FullSimp,
        testset,
    );
    bench_setup(
        "BssWithCats",
        MAX_TCOUNT,
        &BssWithCatsDriver { random_t: false },
        SimpFunc::FullSimp,
        testset,
    );
    bench_setup(
        "DynamicT",
        MAX_TCOUNT,
        &DynamicTDriver,
        SimpFunc::FullSimp,
        testset,
    );
}

// fn benchmark_simplifier(testset: &Vec<VecGraph>) {
//     println!("Running Simplifier benchmarks...");

//     bench_setup("NoSimp", 12, &BssWithCatsDriver { random_t:false }, SimpFunc::NoSimp, testset);
//     bench_setup("CliffSimp", MAX_TCOUNT, &BssWithCatsDriver { random_t:false }, SimpFunc::CliffordSimp, testset);
//     bench_setup("FullSimp", MAX_TCOUNT, &BssWithCatsDriver { random_t:false }, SimpFunc::FullSimp, testset);
// }

pub fn bench() -> Result<(), CliError> {
    // Create results directory if it doesn't exist
    std::fs::create_dir_all("benches/results").expect("Failed to create results directory");

    println!("Generating test set...");
    let testset = get_testset();
    assert_eq!(
        testset.len(),
        (MAX_TCOUNT - MIN_TCOUNT) * SAMPLES_PER_TCOUNT
    );

    // Run benchmarks
    benchmark_driver(&testset);
    // benchmark_simplifier(&testset);

    // Generate plots
    println!("Generating plots...");

    // Find all alpha (nterms) files
    let alpha_files: Vec<String> = std::fs::read_dir("benches/results")
        .unwrap()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.to_str()?.contains("benchmark_alpha_") {
                Some(path.to_str()?.to_string())
            } else {
                None
            }
        })
        .collect();

    // Find all times files
    let times_files: Vec<String> = std::fs::read_dir("benches/results")
        .unwrap()
        .filter_map(|entry| {
            let path = entry.ok()?.path();
            if path.to_str()?.contains("benchmark_times_") {
                Some(path.to_str()?.to_string())
            } else {
                None
            }
        })
        .collect();

    // Create plots
    create_svg_plot("alpha", alpha_files, "benches/results/nterms_plot.svg")
        .expect("Failed to create nterms plot");
    create_svg_plot("times", times_files, "benches/results/runtime_plot.svg")
        .expect("Failed to create runtime plot");

    println!("Benchmarking complete! Plots saved to benches/results/");
    Ok(())
}
