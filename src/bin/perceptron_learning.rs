// Experiment with learning AND, OR, and XOR gates (two-dimensional input).
// Show the learning process using graphs (two-dimensional straight-line graph).
// Error graph for iterative learning
// Implement using modules // Bonus points if implemented as a class.
// Compose output calculation and learning process as member functions.
const SAMPLE_SIZE: u32 = 4;

use std::thread::sleep;
use std::time::Duration;

//AND Gate
fn main(){
    let mut x = load_x();

    //OR GATE
    println!("OR GATE");
    let y: Vec<i32> = vec![0,1,1,1];
    learning(&x, &y);
    println!("/////////////");
    println!("");

    //AND GATE
    println!("AND GATE");
    let y: Vec<i32> = vec![0,0,0,1];
    learning(&x, &y);
    println!("/////////////");

    //XOR GATE
    // let y: Vec<i32> = vec![0,1,1,0];
    // learning(&x, &y);

}

//Get input samples
fn load_x() -> Vec<Vec<i32>>{
    vec![vec![0,0], vec![0,1],vec![1,0],vec![1,1]]
}

//Learning -> get weights
fn learning(x: &Vec<Vec<i32>>, y: &Vec<i32>){
    let mut weights = init_weight(x[0].len() as usize);
    let mut bias: f64 = 0.5;
    let learning_rate = 0.5;

    let mut iter_count = 0;

    let a = forward_propagation(x, &weights, bias);
    let mut o = classification(&a);
    println!("init output: {:?} //// w1:{:.3}, w2:{:.3}", o, weights[0], weights[1]);

    while &o!=y {
        iter_count += 1;

        weights_update(&mut weights, &mut bias, &a, x, y, learning_rate);

        let a = forward_propagation(x, &weights, bias);
        o = classification(&a);
        println!("{:?}th -- {:?} //w1:{:.3}, w2:{:.3}, bias:{:.3}", iter_count, o, weights[0], weights[1], bias);
        sleep(Duration::from_millis(500))
    }
    println!("{:?}th iter => w1:{:.3}, w2:{:.3}, bias:{:.3}", iter_count, weights[0], weights[1], bias);

}

// fn multi_layer_learning(x: &Vec<Vec<i32>>, y: &Vec<i32>, node_num: Vec<i32>){
//     let learning_rate = 0.5;

//     let mut all_weights= Vec::new();
//     let mut all_a = Vec::new();
//     let mut all_b: Vec<f64> = Vec::new();

//     for i in 0..node_num.len(){
//             let mut weights = init_weight(x[0].len() as usize);
//             let mut bias: f64 = 0.0;

//             all_weights.push(weights);
//             all_b.push(bias);
//     }

//     let mut iter_count = 0;

//     //TODO 이거수정
//     let a = forward_propagation(x, &weights, bias);
//     let mut o = classification(&a);
//     println!("init output: {:?} //// w1:{:.3}, w2:{:.3}", o, weights[0], weights[1]);

//     while &o!=y {
//         iter_count += 1;

//         weights_update(&mut weights, &mut bias, &a, x, y, learning_rate);

//         let a = forward_propagation(x, &weights, bias);
//         o = classification(&a);
//         println!("{:?}th -- {:?} //w1:{:.3}, w2:{:.3}, w3:{:.3}", iter_count, o, weights[0], weights[1], weights[2]);
//         sleep(Duration::from_millis(500))
//     }
//     println!("{:?}th iter => w1:{:.3}, w2:{:.3}, b:{:.3}", iter_count, weights[0], weights[1], bias);
// }

// weight initialize
fn init_weight(dim:usize)->Vec<f64>{
    vec![1.1; dim]
}

fn forward_propagation(x: &Vec<Vec<i32>>, weights: &Vec<f64>, bias: f64)->Vec<f64>{
    let mut a_vec: Vec<f64> = vec![];

    for x_sample in x{
        let mut z: f64 = 0.0;

        for node_num in 0..x_sample.len(){
            //int to float
            z += (x_sample[node_num] as f64) * weights[node_num]
        }
        z += bias;
        let a = sigmoid(z);
        a_vec.push(a);
    }

    a_vec
}

fn weights_update(weights: &mut Vec<f64>, bias: &mut f64, a: &Vec<f64> ,x: &Vec<Vec<i32>>, y: &Vec<i32>, alpha: f64)->Vec<f64>{
    let mut dW_vec = vec![];
    let mut db: f64 = 0.0;

    for i in 0..weights.len(){
        let mut dW = 0.0;

        for j in 0..x.len(){
            dW += alpha * (a[i] - (y[i] as f64)) * a[i] * (1.0 - a[i]) * (x[j][i] as f64);
            db += alpha * (a[i] - (y[i] as f64)) * a[i] * (1.0 - a[i]);
        } 
        dW = dW/(SAMPLE_SIZE as f64);
        db = db/(SAMPLE_SIZE as f64);

        dW_vec.push(dW);
        weights[i] = weights[i] - dW;
        *bias = *bias - db;
    }
    dW_vec
}

// Output -> 0 or 1
fn classification(a_vec: &Vec<f64>)->Vec<i32>{
    let mut output: Vec<i32> = vec![];
    for a in a_vec{
        if *a > 0.5 {
            output.push(1);
        }
        else{ 
            output.push(0); 
        }
    }
    output
}

//sigmoid function
fn sigmoid(z: f64)->f64{
    1.0/(1.0+(-z).exp())
}

