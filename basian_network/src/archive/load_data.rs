use std::fs::File;
use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use std::thread;
use csv::ReaderBuilder;
use csv::StringRecord;
use dashmap::DashMap;
use std::error::Error;


pub fn generate_contingency_table(path: &str) -> Result<(Arc<DashMap<Vec<u8>, u32>>, HashMap<String, HashMap<String, u8>>, HashMap<u8, String>), Box<dyn Error>> {
    let file = File::open(path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let headers: Vec<String> = rdr.headers()?.iter().map(|s| s.to_string()).collect();
    let mut data = Vec::new();
    let mut category_maps: Vec<HashMap<String, u8>> = vec![HashMap::new(); headers.len()];

    // Read data and build category maps
    for result in rdr.records() {
        let record = result?;
        let row: Vec<String> = record.iter().map(|s| s.to_string()).collect();
        data.push(row.iter().enumerate().map(|(i, value)| {
            let map = &mut category_maps[i];
            let len = map.len();
            *map.entry(value.clone()).or_insert_with(|| len as u8)
        }).collect::<Vec<u8>>());
    }

    let data = Arc::new(Mutex::new(data));
    let freq_map: Arc<DashMap<Vec<u8>, u32>> = Arc::new(DashMap::new());
    let num_cpus = num_cpus::get();
    let chunk_size = (data.lock().unwrap().len() + num_cpus - 1) / num_cpus; // 20000

    let handles: Vec<_> = (0..num_cpus).map(|i| {
        let data = Arc::clone(&data);
        let freq_map = Arc::clone(&freq_map);
        thread::spawn(move || {
            let data = data.lock().unwrap();
            let start = i * chunk_size;
            let end = std::cmp::min((i + 1) * chunk_size, data.len());

            for row in &data[start..end] {
                freq_map.entry(row.clone()).and_modify(|e| *e += 1).or_insert(1);
            }
        })
    }).collect();

    for handle in handles {
        handle.join().unwrap();
    }
    // Convert headers to index to name map
    let headers_index_map: HashMap<u8, String> = headers.iter().enumerate().map(|(i, h)| (i as u8, h.clone())).collect();
    // Prepare header to category map
    let header_to_category_map: HashMap<String, HashMap<String, u8>> = headers.clone().into_iter()
        .zip(category_maps.into_iter())
        .collect();
    Ok((freq_map, header_to_category_map, headers_index_map))
}
