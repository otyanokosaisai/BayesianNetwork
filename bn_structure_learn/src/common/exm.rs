use crate::common;

pub fn exm_order() {
    let setting_path = "./setting/exm/order.yml";
    let mut data_container = match common::core::order_base::load_data(setting_path) {
        Ok(data_container) => data_container,
        Err(e) => panic!("{:?}", e),
    };
    data_container.learn();
    data_container.visualize();
    data_container.evaluate();
    data_container.compare();
    // data_container.cpdag_visualize();
}

pub fn exm_score() {
    let setting_path = "./setting/exm/score.yml";
    let mut data_container = match common::core::score_base::load_data(&setting_path) {
        Ok(data_container) => data_container,
        Err(e) => panic!("{:?}", e),
    };
    data_container.analyze();
    data_container.visualize();
    data_container.evaluate();
    data_container.compare();
}