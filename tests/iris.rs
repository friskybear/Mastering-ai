use mastering_ai::{
    Activation, DenseLayer, LayerNeuron, Loss, LossFunction, MLP, SGD, Vector, normalize_dataset,
    normalize_sample, shuffle, train_test_split,
};

// Strip surrounding double-quotes from a CSV field if present.
fn strip_quotes(s: &str) -> &str {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') {
        &s[1..s.len() - 1]
    } else {
        s
    }
}

// One-hot encode the iris variety label.
fn parse_label(label: &str) -> Vector {
    match label {
        "Setosa" => Vector::new(vec![1.0, 0.0, 0.0]),
        "Versicolor" => Vector::new(vec![0.0, 1.0, 0.0]),
        "Virginica" => Vector::new(vec![0.0, 0.0, 1.0]),
        other => panic!("unknown label: {:?}", other),
    }
}

// Load the iris CSV.
// Returns (feature_names, Vec<(features, one_hot_label)>).
fn load_iris(path: &str) -> (Vec<String>, Vec<(Vector, Vector)>) {
    let contents =
        std::fs::read_to_string(path).unwrap_or_else(|e| panic!("failed to read {}: {}", path, e));
    let mut lines = contents.lines();

    // Header row
    let header = lines.next().expect("CSV has no header");
    let feature_names: Vec<String> = header
        .split(',')
        .take(4)
        .map(|s| strip_quotes(s).to_string())
        .collect();

    // Data rows
    let data = lines
        .filter(|l| !l.trim().is_empty())
        .map(|line| {
            let parts: Vec<&str> = line.split(',').collect();
            assert!(
                parts.len() >= 5,
                "expected >=5 columns, got {} in line: {:?}",
                parts.len(),
                line
            );
            let features = Vector::from(
                parts[..4]
                    .iter()
                    .map(|v| {
                        strip_quotes(v)
                            .parse::<f64>()
                            .unwrap_or_else(|_| panic!("bad float: {:?}", v))
                    })
                    .collect::<Vec<f64>>(),
            );
            let label = parse_label(strip_quotes(parts[4]));
            (features, label)
        })
        .collect();

    (feature_names, data)
}

// Argmax over a Vector's values.
fn argmax(v: &Vector) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap()
}

#[test]
fn iris() {
    // ------------------------------------------------------------------
    // 1. Load & shuffle
    // ------------------------------------------------------------------
    let (_feature_names, mut dataset) = load_iris("asset/iris.csv");
    assert_eq!(dataset.len(), 150, "iris should have 150 samples");

    // Randomly shuffle before splitting so each class is represented in
    // both train and test regardless of the original ordering in the CSV.
    shuffle(&mut dataset);

    // ------------------------------------------------------------------
    // 2. Split 80 / 20  (120 train, 30 test)
    // ------------------------------------------------------------------
    let (train_raw, test_raw) = train_test_split(&dataset, 0.80);

    println!("Split: {} train, {} test", train_raw.len(), test_raw.len());

    // ------------------------------------------------------------------
    // 3. Normalize features to [0, 1] using TRAIN statistics only.
    //    Applying test data to the same scale avoids data leakage.
    // ------------------------------------------------------------------
    let (mut train_features, train_labels): (Vec<Vector>, Vec<Vector>) =
        train_raw.into_iter().unzip();

    // Fit normalizer on train set; returns (mins, maxs) per feature.
    let (mins, maxs) = normalize_dataset(&mut train_features);

    let train_data: Vec<(Vector, Vector)> = train_features.into_iter().zip(train_labels).collect();

    // Apply the same transform to test features.
    let (test_features, test_labels): (Vec<Vector>, Vec<Vector>) = test_raw.into_iter().unzip();

    let test_data: Vec<(Vector, Vector)> = test_features
        .into_iter()
        .map(|mut f| {
            normalize_sample(&mut f, &mins, &maxs);
            f
        })
        .zip(test_labels)
        .collect();

    // ------------------------------------------------------------------
    // 4. Build MLP   4 -> 8 (Tanh) -> 3 (Sigmoid)
    //    Xavier init throughout (suitable for Tanh + Sigmoid).
    // ------------------------------------------------------------------
    let hidden = DenseLayer::new(vec![
        LayerNeuron::new(Vector::rand_xavier(4, 4, 8), 0.0, Activation::Tanh),
        LayerNeuron::new(Vector::rand_xavier(4, 4, 8), 0.0, Activation::Tanh),
        LayerNeuron::new(Vector::rand_xavier(4, 4, 8), 0.0, Activation::Tanh),
        LayerNeuron::new(Vector::rand_xavier(4, 4, 8), 0.0, Activation::Tanh),
        LayerNeuron::new(Vector::rand_xavier(4, 4, 8), 0.0, Activation::Tanh),
        LayerNeuron::new(Vector::rand_xavier(4, 4, 8), 0.0, Activation::Tanh),
        LayerNeuron::new(Vector::rand_xavier(4, 4, 8), 0.0, Activation::Tanh),
        LayerNeuron::new(Vector::rand_xavier(4, 4, 8), 0.0, Activation::Tanh),
    ]);

    let output = DenseLayer::new(vec![
        LayerNeuron::new(Vector::rand_xavier(8, 8, 3), 0.0, Activation::Sigmoid),
        LayerNeuron::new(Vector::rand_xavier(8, 8, 3), 0.0, Activation::Sigmoid),
        LayerNeuron::new(Vector::rand_xavier(8, 8, 3), 0.0, Activation::Sigmoid),
    ]);

    let mut mlp = MLP::new(vec![hidden, output]);
    let loss_fn = LossFunction::new(Loss::MSE);
    let mut optimizer = SGD::new(0.05).with_momentum(0.9).with_decay(1e-4);

    // ------------------------------------------------------------------
    // 5. Train on the 80% split
    // ------------------------------------------------------------------
    for epoch in 0..=20_000 {
        mlp.train_batch(&train_data, &loss_fn, &mut optimizer);

        if epoch % 2_000 == 0 {
            let train_loss = LossFunction::batch_loss(
                &loss_fn,
                &train_data
                    .iter()
                    .map(|(x, _)| mlp.forward(x).1)
                    .collect::<Vec<_>>(),
                &train_data
                    .iter()
                    .map(|(_, y)| y.clone())
                    .collect::<Vec<_>>(),
            );
            println!("epoch {:6} | train loss {:.6}", epoch, train_loss);
        }
    }

    // ------------------------------------------------------------------
    // 6. Evaluate on the held-out 20% test split
    // ------------------------------------------------------------------
    let train_correct = train_data
        .iter()
        .filter(|(x, y)| {
            let (_, out) = mlp.forward(x);
            argmax(&out) == argmax(y)
        })
        .count();

    let test_correct = test_data
        .iter()
        .filter(|(x, y)| {
            let (_, out) = mlp.forward(x);
            argmax(&out) == argmax(y)
        })
        .count();

    let train_acc = train_correct as f64 / train_data.len() as f64;
    let test_acc = test_correct as f64 / test_data.len() as f64;

    println!(
        "\nTrain accuracy: {}/{} = {:.1}%",
        train_correct,
        train_data.len(),
        train_acc * 100.0
    );
    println!(
        "Test  accuracy: {}/{} = {:.1}%",
        test_correct,
        test_data.len(),
        test_acc * 100.0
    );

    // Iris is mostly linearly separable; a well-trained MLP should generalise
    // to the unseen 20% with at least 90% accuracy.
    assert!(
        test_acc >= 0.90,
        "expected test accuracy >= 90%, got {:.1}%",
        test_acc * 100.0
    );
}
