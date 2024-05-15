mod score_base;
mod order_base;
use anyhow::Result;
use std::env;


fn main() -> Result<()> {
    //引数がorderならorder_baseを実行
    let args: Vec<String> = env::args().collect();
    if args.len() > 1 && args[1] == "order" {
        let setting_path = "./setting/setting_order.yml";
        let mut data_container = match order_base::load_data(setting_path) {
            Ok(data_container) => data_container,
            Err(e) => panic!("{:?}", e),
        };
        data_container.learn();
        data_container.visualize();
        data_container.evaluate();
        data_container.compare();
        data_container.cpdag_visualize();
    } else if args.len() > 1 && args[1] == "score" {
        let setting_path = "./setting/setting_score.yml";
        let mut data_container = match score_base::load_data(&setting_path) {
            Ok(data_container) => data_container,
            Err(e) => panic!("{:?}", e),
        };
        data_container.analyze();
        data_container.visualize();
        data_container.evaluate();
        data_container.compare();
        println!("header: {:?}", data_container.ct.header);
        println!("network: {:?}", data_container.network.network_values);
        println!("compnet: {:?}", data_container.compare_network.network_values);
    } else {
        panic!("引数が不正です");
    }
    Ok(())
}