use std::fs::File;
use std::io::Read;
use serde_yaml;
use serde::{Deserialize, Serialize};
use crate::common::core::{order_base, score_base};

use super::core::utils::Setting;

#[derive(Debug, Deserialize, Serialize)]
struct EstSetting {
    pub method: String,
    pub bdeu_ess: f64,
    pub valid_rate: f64,
    pub data_dir: String,
    pub saving_dir: String,
}

fn read_est_setting(setting_path: &str) -> EstSetting {
    let mut file = File::open(setting_path).expect("Failed to open setting file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read setting file");
    let setting: EstSetting = serde_yaml::from_str(&contents).expect(format!("Failed to parse setting file: {}", setting_path).as_str());
    setting
}

pub fn est_order() {
    let setting = read_est_setting("./setting/est/order.yml");
    let setting = Setting {
        method: setting.method,
        bdeu_ess: setting.bdeu_ess,
        valid_rate: setting.valid_rate,
        data_path: setting.data_dir,
        compare_network_path: "./setting/asia.dot".to_string(), // dummy
        saving_dir: setting.saving_dir,
    };
    let mut data_container = match order_base::load_data_from_setting(setting) {
        Ok(data_container) => data_container,
        Err(e) => panic!("{:?}", e),
    };
    data_container.learn();
    data_container.visualize();
    let score = data_container.evaluate();
    println!("[est_order] {:?}: {:?}", data_container.scoring_method.method, score);
}

pub fn est_score() {
    let setting = read_est_setting("./setting/est/score.yml");
    let setting = Setting {
        method: setting.method,
        bdeu_ess: setting.bdeu_ess,
        valid_rate: setting.valid_rate,
        data_path: setting.data_dir,
        compare_network_path: "./setting/asia.dot".to_string(), //dummy
        saving_dir: setting.saving_dir,
    };
    let mut data_container = match score_base::load_data_from_setting(setting) {
        Ok(data_container) => data_container,
        Err(e) => panic!("{:?}", e),
    };
    data_container.analyze();
    data_container.visualize();
    let score = data_container.evaluate();
    println!("[est_score] {:?}: {:?}", data_container.scoring_method.method, score);
}