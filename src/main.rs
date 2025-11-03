// Copyright (c) 2025, jose cornado

#![allow(non_snake_case)]

mod mlp;
mod network;

use std::{
    sync::mpsc::{self, Receiver, Sender, TryRecvError},
    thread,
    time::{Duration, Instant},
};

use mlp::register::REGISTER_WIDTH;
use network::get_network;

use csv::{ReaderBuilder, StringRecord};

#[tokio::main(flavor = "multi_thread", worker_threads = 4)]
async fn main() {
    let mut rdr = ReaderBuilder::new().from_path("mnist_train.csv").unwrap();   
    let records = rdr.records().collect::<Result<Vec<StringRecord>, csv::Error>>().unwrap();
    let mut mnist_train: Vec<Vec<f32>> = vec![];
    records.iter().for_each(|r|{
        let mut record: Vec<f32> = vec![];
        r.iter().for_each(|item| { record.push(item.parse::<f32>().unwrap()); } );
        mnist_train.push(record);
    });

    let layers_json = "[{
        \"layer\": \"Conv2D\",
        \"filter_count\": 6,
        \"dimension_0\": 28,
        \"dimension_1\": 5,
        \"dimension_2\": 5,
        \"padding\": 2,
        \"stride\": 1
    }]".as_bytes();

    let (sender, receiver) = get_network(layers_json);
    tokio::spawn(async move {
        let mut i = 0;
        let start = Instant::now();
        loop{
            match receiver.try_recv() {
                Ok(message) => {
                     i += 1;
                }
                Err(e) => match e {
                    TryRecvError::Empty => thread::sleep(Duration::from_micros(10)),
                    TryRecvError::Disconnected => {
                        println!("{} millis, {} images, output disconnected", start.elapsed().as_millis(), i);
                        break;
                    }
                }
            }
        }            
            
    });
    while mnist_train.len() != 0 {
        let mut record = mnist_train.pop().unwrap();
        record.remove(0);
        record.resize(
            record.len() + (REGISTER_WIDTH / (size_of::<f32>() * 8))
                - record.len() % (REGISTER_WIDTH / (size_of::<f32>() * 8)),
            0.0,
        );
        _ = sender.send(vec![record]);
    }
    drop(sender);
}
