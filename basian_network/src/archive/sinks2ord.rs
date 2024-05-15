use std::collections::HashMap;

pub fn sinks2ord(v: &Vec<u8>, sink: HashMap<Vec<u8>, u8>) -> Vec<u8> {
    let mut ord = vec![u8::MAX; v.len()]; // ord 配列を初期化
    let mut left = v.clone(); // left を v のクローンとして初期化
    left.sort();
    for i in (0..v.len()).rev() {
        let current_sink = match sink.get(&left) {
            Some(sink) => *sink,
            None => panic!("Sink not found: {:?}", left),
        }; // left に対応する最良の親ノードを取得
        ord[i] = current_sink;
        left.retain(|&x| x != current_sink); // left から選ばれた要素を削除
    }
    // println!("ord: {:?}", ord);
    ord
}