use std::fs::File;
use std::io::Read;
use serde_yaml;
use serde::{Deserialize, Serialize};
use crate::{order_base, score_base};

#[derive(Debug, Deserialize, Serialize)]
struct EstSetting {
    pub method: String,
    pub bdeu_ess: f64,
    pub data_dir: String,
}

fn read_est_setting(setting_path: &str) -> EstSetting {
    let mut file = File::open(setting_path).expect("Failed to open setting file");
    let mut contents = String::new();
    file.read_to_string(&mut contents).expect("Failed to read setting file");
    let setting: EstSetting = serde_yaml::from_str(&contents).expect(format!("Failed to parse setting file: {}", setting_path).as_str());
    setting
}

pub fn est_order() {
    let setting = read_est_setting("./est/setting_order.yml");
    let setting = order_base::Setting {
        method: setting.method,
        bdeu_ess: setting.bdeu_ess,
        data_path: setting.data_dir,
        compare_network_path: "data/asia.dot".to_string(),
    };
    let mut data_container = match order_base::load_data_exp(setting) {
        Ok(data_container) => data_container,
        Err(e) => panic!("{:?}", e),
    };
    data_container.learn();
    data_container.visualize();
    data_container.evaluate();
}

pub fn est_score() {
    let setting = read_est_setting("./est/setting_score.yml");
    let setting = score_base::Setting {
        method: setting.method,
        bdeu_ess: setting.bdeu_ess,
        data_path: setting.data_dir,
        compare_network_path: "data/asia.dot".to_string(),
    };
    let mut data_container = match score_base::load_data_exp(setting) {
        Ok(data_container) => data_container,
        Err(e) => panic!("{:?}", e),
    };
    data_container.analyze();
    data_container.visualize();
    data_container.evaluate();
}