// Experiment with learning AND, OR, and XOR gates (two-dimensional input).
// Show the learning process using graphs (two-dimensional straight-line graph).
// Error graph for iterative learning
// Implement using modules // Bonus points if implemented as a class.
// Compose output calculation and learning process as member functions.
const INPUT_DIM: u32 = 2;
const SAMPLE_SIZE: u32 = 4;

//AND Gate
fn main(){
    let x = input_x();
    let learning_rate = 1.0;
    let iter = 1;

    //random values
    let mut weights = vec![0.2,0.5];
    let mut bias: f64 = 0.0;

    let a = forward_propagation(&x, &weights, bias);
    let o = classification(&a);
    println!("{:?}", o);

    weights_update(&mut weights, &a, &x, learning_rate);
    println!("weights of {:?}th iter: {:?}", iter,weights);

    let a = forward_propagation(&x, &weights, bias);
    let o = classification(&a);
    println!("{:?}", o);


    weights_update(&mut weights, &a, &x, learning_rate);
    println!("weights of {:?}th iter: {:?}", iter,weights);

    let a = forward_propagation(&x, &weights, bias);
    let o = classification(&a);
    println!("{:?}", o);

}

fn input_x() -> Vec<Vec<i32>>{
    vec![vec![0,0], vec![0,1],vec![1,0],vec![1,1]]
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

fn weights_update(weights: &mut Vec<f64>, a: &Vec<f64> ,x: &Vec<Vec<i32>>, alpha: f64){
    for i in 0..x[0].len(){
        let mut dW = 0.0;
        for j in 0..x.len(){
            dW += alpha * a[i] * (1.0 - a[i]) * (x[j][i] as f64);
        } 
        dW = dW/(SAMPLE_SIZE as f64);
        weights[i] = weights[i] - dW;
    }
}

fn sigmoid(z: f64)->f64{
    1.0/(1.0+(-z).exp())
}


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
