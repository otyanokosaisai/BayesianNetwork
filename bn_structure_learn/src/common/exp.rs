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
    pub valid_rate: f64,
    pub data_dir: String,
    pub saving_dir: String,
    pub compare_network_dir: String,
    pub data_names: Vec<String>,
    pub timeout_hours: u64,
    pub timeout_valid: bool,
    pub trial_num: usize,
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

pub fn exp_order() {
    let exp_setting = read_exp_setting("./setting/exp/order.yml");
    let timeout_duration = Duration::from_secs(exp_setting.timeout_hours * 3600);
    let log_file_path = format!("{}/{}.txt", exp_setting.saving_dir, current_timestamp());
    println!("logs: {}", log_file_path);
    let running = Arc::new((Mutex::new(true), Condvar::new()));

    'data_name_loop: for data_name in exp_setting.data_names {
        for ess in &exp_setting.bdeu_ess { // 各 bdeu_ess に対してループ
            for trial in 0..exp_setting.trial_num { // trial_num の回数分実行
                let save_dir = format!("{}/{}/ess_{}_trial_{}", exp_setting.saving_dir, data_name, ess, trial);
                // もしディレクトリが存在しないなら作成
                if !std::path::Path::new(&save_dir).exists() {
                    std::fs::create_dir_all(&save_dir).expect("Failed to create directory");
                }
                let setting = Setting {
                    method: exp_setting.method.clone(),
                    bdeu_ess: *ess,
                    valid_rate: exp_setting.valid_rate,
                    data_path: format!("{}/{}.csv", exp_setting.data_dir, data_name),
                    compare_network_path: format!("{}/{}.dot", exp_setting.compare_network_dir, data_name),
                    saving_dir: save_dir,
                };

                // タイマー用のスレッドを作成
                let start_time = Instant::now();
                let running_clone = running.clone();
                let data_name_clone = data_name.clone();
                let log_file_path_clone = log_file_path.clone();

                let timeout_thread = thread::spawn(move || {
                    let (lock, cvar) = &*running_clone;
                    let mut run = lock.lock().unwrap();
                    while *run {
                        let result = cvar.wait_timeout(run, timeout_duration).unwrap();
                        run = result.0;
                        if result.1.timed_out() {
                            *run = false;
                            let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path_clone).expect("Cannot open log file");
                            writeln!(log_file, "Process for {} timed out.", data_name_clone).expect("Failed to write to log file");
                            break;
                        }
                    }
                });

                let mut data_container = order_base::load_data_from_setting(setting).expect("error");

                // メインの実験処理
                if *running.0.lock().unwrap() && start_time.elapsed() < timeout_duration {
                    let learn_start = Instant::now();
                    data_container.learn();
                    let learn_duration = learn_start.elapsed();

                    let visualize_start = Instant::now();
                    data_container.visualize();
                    let visualize_duration = visualize_start.elapsed();

                    let evaluate_start = Instant::now();
                    data_container.evaluate();
                    let evaluate_duration = evaluate_start.elapsed();

                    let compare_start = Instant::now();
                    let output_evaluate = data_container.compare_all();
                    let compare_duration = compare_start.elapsed();

                    // 結果をログファイルに書き込む
                    let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path).expect("Cannot open log file");
                    writeln!(log_file, "Experiment for {} with ess {} trial {} completed successfully.", data_name, ess, trial).expect("Failed to write to log file");
                    writeln!(log_file, "learn duration: {:?}", learn_duration).expect("Failed to write to log file");
                    writeln!(log_file, "visualize duration: {:?}", visualize_duration).expect("Failed to write to log file");
                    writeln!(log_file, "evaluate duration: {:?}", evaluate_duration).expect("Failed to write to log file");
                    for text in output_evaluate {
                        writeln!(log_file, "{}", text).expect("Failed to write to log file");
                    }
                    writeln!(log_file, "compare duration: {:?}", compare_duration).expect("Failed to write to log file");

                    // 実験処理が完了したら、タイマー用スレッドを通知して終了させる
                    let (lock, cvar) = &*running;
                    let mut run = lock.lock().unwrap();
                    *run = false;
                    cvar.notify_all();
                } else {
                    if !*running.0.lock().unwrap() {
                        let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path).expect("Cannot open log file");
                        writeln!(log_file, "Experiment for {} with ess {} trial {} was stopped after reaching the timeout limit.", data_name, ess, trial).expect("Failed to write to log file");
                        timeout_thread.join().unwrap();
                        if exp_setting.timeout_valid {
                            break 'data_name_loop; // timeout_valid が true の場合、データセットループを終了
                        } else {
                            break; // timeout_valid が false の場合、trial_num のループを終了
                        }
                    }
                }

                // タイマー用スレッドの終了を待機
                timeout_thread.join().unwrap();

                // 次のデータセットの処理を開始する前に running を true に戻す
                let mut run = running.0.lock().unwrap();
                *run = true;

                // タイムアウトが発生した場合、次のデータセットに移動
                if !*run && exp_setting.timeout_valid {
                    break 'data_name_loop;
                }
            }
        }
    }
}

pub fn exp_score() {
    let exp_setting = read_exp_setting("./setting/exp/score.yml");
    let timeout_duration = Duration::from_secs(exp_setting.timeout_hours * 3600);
    let log_file_path = format!("{}/{}.txt", exp_setting.saving_dir, current_timestamp());
    println!("logs: {}", log_file_path);
    let running = Arc::new((Mutex::new(true), Condvar::new()));

    'data_name_loop: for data_name in exp_setting.data_names {
        for ess in &exp_setting.bdeu_ess { // 各 bdeu_ess に対してループ
            for trial in 0..exp_setting.trial_num { // trial_num の回数分実行
                let save_dir = format!("{}/{}/ess_{}_trial_{}", exp_setting.saving_dir, data_name, ess, trial);
                // もしディレクトリが存在しないなら作成
                if !std::path::Path::new(&save_dir).exists() {
                    std::fs::create_dir_all(&save_dir).expect("Failed to create directory");
                }
                let setting = Setting {
                    method: exp_setting.method.clone(),
                    bdeu_ess: *ess,
                    valid_rate: exp_setting.valid_rate,
                    data_path: format!("{}/{}.csv", exp_setting.data_dir, data_name),
                    compare_network_path: format!("{}/{}.dot", exp_setting.compare_network_dir, data_name),
                    saving_dir: save_dir,
                };

                // タイマー用のスレッドを作成
                let start_time = Instant::now();
                let running_clone = running.clone();
                let data_name_clone = data_name.clone();
                let log_file_path_clone = log_file_path.clone();

                let timeout_thread = thread::spawn(move || {
                    let (lock, cvar) = &*running_clone;
                    let mut run = lock.lock().unwrap();
                    while *run {
                        let result = cvar.wait_timeout(run, timeout_duration).unwrap();
                        run = result.0;
                        if result.1.timed_out() {
                            *run = false;
                            let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path_clone).expect("Cannot open log file");
                            writeln!(log_file, "Process for {} timed out.", data_name_clone).expect("Failed to write to log file");
                            break;
                        }
                    }
                });

                let mut data_container = score_base::load_data_from_setting(setting).expect("error");

                // メインの実験処理
                if *running.0.lock().unwrap() && start_time.elapsed() < timeout_duration {
                    let analyze_start = Instant::now();
                    data_container.analyze();
                    let analyze_duration = analyze_start.elapsed();

                    let visualize_start = Instant::now();
                    data_container.visualize();
                    let visualize_duration = visualize_start.elapsed();

                    let evaluate_start = Instant::now();
                    data_container.evaluate();
                    let evaluate_duration = evaluate_start.elapsed();

                    let compare_start = Instant::now();
                    let output = data_container.compare_all();
                    let compare_duration = compare_start.elapsed();

                    // 結果をログファイルに書き込む
                    let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path).expect("Cannot open log file");
                    writeln!(log_file, "Experiment for {} with ess {} trial {} completed successfully.", data_name, ess, trial).expect("Failed to write to log file");
                    writeln!(log_file, "analyze duration: {:?}", analyze_duration).expect("Failed to write to log file");
                    writeln!(log_file, "visualize duration: {:?}", visualize_duration).expect("Failed to write to log file");
                    writeln!(log_file, "evaluate duration: {:?}", evaluate_duration).expect("Failed to write to log file");
                    for text in output {
                        writeln!(log_file, "{}", text).expect("Failed to write to log file");
                    }
                    writeln!(log_file, "compare duration: {:?}\n", compare_duration).expect("Failed to write to log file");

                    // 実験処理が完了したら、タイマー用スレッドを通知して終了させる
                    let (lock, cvar) = &*running;
                    let mut run = lock.lock().unwrap();
                    *run = false;
                    cvar.notify_all();
                } else {
                    if !*running.0.lock().unwrap() {
                        let mut log_file = OpenOptions::new().append(true).create(true).open(&log_file_path).expect("Cannot open log file");
                        writeln!(log_file, "Experiment for {} with ess {} trial {} was stopped after reaching the timeout limit.", data_name, ess, trial).expect("Failed to write to log file");
                        timeout_thread.join().unwrap();
                        if exp_setting.timeout_valid {
                            break 'data_name_loop; // timeout_valid が true の場合、データセットループを終了
                        } else {
                            break; // timeout_valid が false の場合、trial_num のループを終了
                        }
                    }
                }

                // タイマー用スレッドの終了を待機
                timeout_thread.join().unwrap();

                // 次のデータセットの処理を開始する前に running を true に戻す
                let mut run = running.0.lock().unwrap();
                *run = true;

                // タイムアウトが発生した場合、次のデータセットに移動
                if !*run && exp_setting.timeout_valid {
                    break 'data_name_loop;
                }
            }
        }
    }
}
