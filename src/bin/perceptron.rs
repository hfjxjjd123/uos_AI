// Implement a perceptron with n input dimensions. (one node)
// Initialize the weight values to random values.
// Repeat the following.
// Find the output by using the input of the previous AND gate.
// Find the number of times the output is incorrect for the four inputs.
// Terminate if the output is correct for all input.
// Input a random value and substitute it for each weight.

use std::vec;
use rand::Rng;

fn main(){
    // the number of samples
    const N: usize = 4; 
    // the dimension of vector of X
    const R: usize = 2;

    // set X and Y
    let x: Vec<Vec<i32>> = vec![vec![0,0], vec![0,1],vec![1,0],vec![1,1]];
    let y: Vec<i32> = vec![0,0,0,1];

    // set parameter, bias is fixed now
    let mut weight: Vec<f64>= vec![0.0; R];
    let bias: f64 = 0.5;

    let mut count: u32 = 0;

    //initialize weight vector
    update_weight(&mut weight, R);

    // start learning
    loop{
        let a: Vec<i32> = get_activation(&x, &weight, N.try_into().unwrap(), bias);
        println!("after learning: {:?}/4 - {:?}th", get_correct_num(&a, &y), count);
        println!("{:?}", a);

        if a == y { break; }
        else {
            count+=1;
            update_weight(&mut weight, R);
        }
    }
    println!("{:?}", count);
    println!("result: {:?}", weight);
}

//weight를 0~1의 수로 결정
fn update_weight(weight: &mut Vec<f64>, dim: usize){
    weight.clear();
    for i in 0..dim{
        let mut rng = rand::thread_rng();
        weight.push(rng.gen::<f64>());
    }
}

// findout WX > b in a step
fn get_activation(input: &Vec<Vec<i32>>, weight: &Vec<f64>, n: i32, bias: f64)->Vec<i32>{
    let mut a: Vec<i32> = Vec::new();
    for i in 0..n{
        let index = i as usize;
        let z = input[index][0] as f64 * weight[0] + input[index][1] as f64 * weight[1];
        a.push(if z>bias {1} else {0});
    }
    a
}

// check the num of right sample  in a step
fn get_correct_num(a: &Vec<i32>, y:&Vec<i32>)->u32{
    let mut count:u32 = 0;
    for i in 0..y.len(){
        if a[i] == y[i]{
            count +=1;
        }
    }
    count

}