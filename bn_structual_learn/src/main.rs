mod common;
use anyhow::Result;
use std::env;


fn main() -> Result<()> {
    //引数がorderならorder_baseを実行
    let args: Vec<String> = env::args().collect();
    if args.len() > 2 && args[1] == "exm" && args[2] == "order"{
        common::exm::exm_order();
    } else if args.len() > 2 && args[1] == "exm" && args[2] == "score" {
        common::exm::exm_score();
    } else if args.len() > 2 && args[1] == "exp" && args[2] == "score" {
        common::exp::exp_score();
    } else if args.len() > 2 && args[1] == "exp" && args[2] == "order" {
        common::exp::exp_order();
    } else if args.len() > 2 && args[1] == "est" && args[2] == "score" {
        common::est::est_score();
    } else if args.len() > 2 && args[1] == "est" && args[2] == "order" {
        common::est::est_order();
    } else {
        panic!("引数が不正です");
    }
    Ok(())
}