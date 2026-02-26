use mastering_ai::{Activation, DenseLayer, LayerNeuron, Loss, LossFunction, MLP, SGD, Vector};



#[test]
fn xor() {
    let xor_dataset = || -> Vec<(Vector, Vector)> {
        vec![
            (Vector::from(vec![0.0, 0.0]), Vector::from(vec![0.0])),
            (Vector::from(vec![0.0, 1.0]), Vector::from(vec![1.0])),
            (Vector::from(vec![1.0, 0.0]), Vector::from(vec![1.0])),
            (Vector::from(vec![1.0, 1.0]), Vector::from(vec![0.0])),
        ]
    };
    // Tanh hidden layer: Xavier init (fan_in=2, fan_out=4)
    // Tanh is symmetric around 0 and has no dying-neuron problem unlike ReLU
    let hidden_layer = DenseLayer::new(vec![
        LayerNeuron::new(Vector::rand_xavier(2, 2, 4), 0.0, Activation::Tanh),
        LayerNeuron::new(Vector::rand_xavier(2, 2, 4), 0.0, Activation::Tanh),
        LayerNeuron::new(Vector::rand_xavier(2, 2, 4), 0.0, Activation::Tanh),
        LayerNeuron::new(Vector::rand_xavier(2, 2, 4), 0.0, Activation::Tanh),
    ]);
    // Sigmoid output layer: Xavier init (fan_in=4, fan_out=1)
    let output_layer = DenseLayer::new(vec![LayerNeuron::new(
        Vector::rand_xavier(4, 4, 1),
        0.0,
        Activation::Sigmoid,
    )]);
    let mut mlp = MLP::new(vec![hidden_layer, output_layer]);
    let loss_fn = LossFunction::new(Loss::MSE);
    // lr=0.05: low enough to avoid oscillation on the batch-of-4 XOR loss surface
    let mut optimizer = SGD::new(0.05);
    let data = xor_dataset();

    for epoch in 0..=20_000 {
        mlp.train_batch(&data, &loss_fn, &mut optimizer);

        if epoch % 2000 == 0 {
            let loss = LossFunction::batch_loss(
                &loss_fn,
                &data
                    .iter()
                    .map(|(x, _)| mlp.forward(x).1)
                    .collect::<Vec<_>>(),
                &data.iter().map(|(_, y)| y.clone()).collect::<Vec<_>>(),
            );

            println!("epoch {:6} | loss {:.6}", epoch, loss);
        }
    }

    println!("\n--- Predictions ---");
    for (x, y) in xor_dataset() {
        let (_, out) = mlp.forward(&x);
        println!(
            "[{:.0}, {:.0}] → {:.4}  (target {:.0})",
            x.as_slice()[0],
            x.as_slice()[1],
            out.as_slice()[0],
            y.as_slice()[0],
        );
    }
}
