use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use std::thread;
use std::sync::{Arc, Mutex, Condvar};
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use serde_yaml;
use serde::{Deserialize, Serialize};
use crate::common::core::utils::Setting;
use crate::common::core::{order_base, score_base};

#[derive(Debug, Deserialize, Serialize)]
struct ExpSetting {
    pub method: String,
    pub bdeu_ess: Vec<f64>,
    pub data_dir: String,
    pub saving_dir: String,
    pub compare_network_dir: String,
    pub data_names: Vec<String>,
    pub timeout_hours: u64,
    pub timeout_down: bool
}

fn read_exp_setting(setting_path: &str) -> ExpSetting {
    let mut file = File::open(setting_path).expect("Failed to open setting file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read setting file");
    let setting: ExpSetting = serde_yaml::from_str(&contents).expect("Failed to parse setting file");
    setting
}

fn current_timestamp() -> String {
    let start = SystemTime::now();
    let since_the_epoch = start.duration_since(UNIX_EPOCH).expect("Time went backwards");
    format!("{}", since_the_epoch.as_secs())
}

fn run_experiment<F>(exp_setting: &ExpSetting, log_file_path: &str, data_name: &str, bdeu_value: f64, running: Arc<(Mutex<bool>, Condvar)>, experiment_func: F)
where
    F: Fn(Setting) -> Result<(), Box<dyn std::error::Error>>,
{
    let timeout_duration = Duration::from_secs(exp_setting.timeout_hours * 3600);
    let save_dir = format!("{}/{}", exp_setting.saving_dir, data_name);

    if !std::path::Path::new(&save_dir).exists() {
        std::fs::create_dir_all(&save_dir).expect("Failed to create directory");
    }

    let setting = Setting {
        method: exp_setting.method.clone(),
        bdeu_ess: bdeu_value,
        data_path: format!("{}/{}.csv", exp_setting.data_dir, data_name),
        compare_network_path: format!("{}/{}.dot", exp_setting.compare_network_dir, data_name),
        saving_dir: save_dir,
    };

    let start_time = Instant::now();
    let running_clone = running.clone();
    let data_name_clone = data_name.to_string();
    let log_file_path_clone = log_file_path.to_string();

    let timeout_thread = thread::spawn(move || {
        let (lock, cvar) = &*running_clone;
        let mut run = lock.lock().unwrap();
        while *run {
            let result = cvar.wait_timeout(run, timeout_duration).unwrap();
            run = result.0;
            if result.1.timed_out() {
                *run = false;
                let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path_clone).expect("Cannot open log file");
                writeln!(log_file, "Process for {} with BDeu {} timed out.", data_name_clone, bdeu_value).expect("Failed to write to log file");
                break;
            }
        }
    });

    if *running.0.lock().unwrap() && start_time.elapsed() < timeout_duration {
        if let Err(e) = experiment_func(setting) {
            let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path).expect("Cannot open log file");
            writeln!(log_file, "Error in experiment for {} with BDeu {}: {:?}", data_name, bdeu_value, e).expect("Failed to write to log file");
        }

        let (lock, cvar) = &*running;
        let mut run = lock.lock().unwrap();
        *run = false;
        cvar.notify_all();
    } else {
        if !*running.0.lock().unwrap() {
            let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path).expect("Cannot open log file");
            writeln!(log_file, "Experiment for {} with BDeu {} was stopped after reaching the timeout limit.", data_name, bdeu_value).expect("Failed to write to log file");
        }
    }

    timeout_thread.join().unwrap();
}

pub fn exp_order() {
    let exp_setting = read_exp_setting("./setting/exp/order.yml");
    let log_file_path = format!("{}/{}.txt", exp_setting.saving_dir, current_timestamp());
    println!("logs: {}", log_file_path);
    let running = Arc::new((Mutex::new(true), Condvar::new()));

    for data_name in &exp_setting.data_names {
        for &bdeu_value in &exp_setting.bdeu_ess {
            run_experiment(
                &exp_setting,
                &log_file_path,
                data_name,
                bdeu_value,
                running.clone(),
                |setting| {
                    let mut data_container = order_base::load_data_exp(setting)?;
                    data_container.learn();
                    data_container.visualize();
                    data_container.evaluate();
                    let output_evaluate = data_container.compare_all();
                    data_container.cpdag_visualize();

                    let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path).expect("Cannot open log file");
                    writeln!(log_file, "Experiment for {} with BDeu {} completed successfully.", data_name, bdeu_value).expect("Failed to write to log file");
                    for text in output_evaluate {
                        writeln!(log_file, "{}", text).expect("Failed to write to log file");
                    }
                    Ok(())
                },
            );

            let timeout_down = exp_setting.timeout_down;
            if !*running.0.lock().unwrap() && timeout_down {
                break;
            }

            let mut run = running.0.lock().unwrap();
            *run = true;
        }
    }
}

pub fn exp_score() {
    let exp_setting = read_exp_setting("./setting/exp/score.yml");
    let log_file_path = format!("{}/{}.txt", exp_setting.saving_dir, current_timestamp());
    println!("logs: {}", log_file_path);
    let running = Arc::new((Mutex::new(true), Condvar::new()));

    for data_name in &exp_setting.data_names {
        for &bdeu_value in &exp_setting.bdeu_ess {
            run_experiment(
                &exp_setting,
                &log_file_path,
                data_name,
                bdeu_value,
                running.clone(),
                |setting| {
                    let mut data_container = score_base::load_data_exp(setting)?;
                    data_container.analyze();
                    data_container.visualize();
                    data_container.evaluate();
                    let output = data_container.compare_all();

                    let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path).expect("Cannot open log file");
                    writeln!(log_file, "Experiment for {} with BDeu {} completed successfully.", data_name, bdeu_value).expect("Failed to write to log file");
                    for text in output {
                        writeln!(log_file, "{}", text).expect("Failed to write to log file");
                    }
                    Ok(())
                },
            );

            let timeout_down = exp_setting.timeout_down;
            if !*running.0.lock().unwrap() && timeout_down {
                break;
            }

            let mut run = running.0.lock().unwrap();
            *run = true;
        }
    }
}