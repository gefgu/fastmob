use std::path::Path;

use fastmob_vis::motif_svg::generate_literature_svgs;

fn main() {
    let output_dir = Path::new("fastmob_vis/assets/motifs");
    match generate_literature_svgs(output_dir) {
        Ok(paths) => {
            for path in paths {
                println!("{}", path.display());
            }
        }
        Err(error) => {
            eprintln!("failed to generate motif SVGs: {error}");
            std::process::exit(1);
        }
    }
}
