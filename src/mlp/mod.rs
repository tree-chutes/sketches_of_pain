// Copyright (c) 2025, tree-chutes

pub mod activation_functions;
pub mod aggregator_functions;
mod conv2d;
mod cross_entropy;
mod identity;
pub mod layers;
mod linear_layer;
pub mod loss_functions;
mod mean_squares;
mod register;
mod relu;
mod sigmoid;
mod softmax;

#[cfg(test)]
mod tests {
    use super::{
        layers::{Layers, layer_factory},
        loss_functions::{LossFunctions, loss_function_factory},
    };
    #[test]
    fn test_conv2d_conv2d_linear_7_5_3_64() {
        let mut input_layer = vec![
            vec![1.0, 0.5, 1.2, 0.8, 1.5, 0.9, 1.3, 0.7, 1.1],
            vec![0.6, 1.4, 0.8, 1.7, 1.0, 1.6, 0.9, 1.2, 0.5],
            vec![1.3, 0.7, 1.8, 1.1, 1.4, 0.6, 1.9, 1.0, 1.5],
            vec![0.9, 1.2, 0.8, 1.6, 1.3, 1.1, 0.7, 1.4, 0.9],
            vec![1.1, 0.8, 1.5, 1.0, 1.7, 1.2, 1.4, 0.6, 1.3],
            vec![0.7, 1.3, 0.9, 1.4, 1.1, 1.8, 1.0, 1.5, 0.8],
            vec![1.2, 0.6, 1.4, 0.9, 1.3, 1.0, 1.6, 1.1, 1.7],
            vec![0.8, 1.1, 0.7, 1.5, 1.2, 1.4, 0.9, 1.3, 1.0],
            vec![1.0, 1.5, 0.8, 1.2, 0.9, 1.3, 1.1, 0.7, 1.4],
        ];

        let conv_0_weights = vec![
            vec![0.1, 0.2, 0.3, 0.2, 0.1],
            vec![0.2, 0.4, 0.6, 0.4, 0.2],
            vec![0.3, 0.6, 0.9, 0.6, 0.3],
            vec![0.2, 0.4, 0.6, 0.4, 0.2],
            vec![0.1, 0.2, 0.3, 0.2, 0.1],
        ];

        let conv_1_weights = vec![
            vec![1.0, 0.5, 0.2],
            vec![0.5, 1.0, 0.5],
            vec![0.2, 0.5, 1.0],
        ];

        let linear_weights: Vec<Vec<f64>> = vec![
            vec![0.1],
            vec![0.2],
            vec![0.3],
            vec![0.4],
            vec![0.5],
            vec![0.6],
            vec![0.7],
            vec![0.8],
            vec![0.9],
        ];

        let linear_bias = vec![vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];

        //Pytorch matches
        let CONV_0_OUTPUT = [
            9.55,
            9.96,
            10.090000000000002,
            9.63,
            9.41,
            9.59,
            10.169999999999998,
            10.270000000000001,
            9.870000000000001,
            9.44,
            9.430000000000001,
            10.01,
            10.4,
            10.110000000000001,
            9.780000000000001,
            9.04,
            9.82,
            10.28,
            10.34,
            9.97,
            8.81,
            9.450000000000001,
            9.930000000000003,
            10.100000000000001,
            10.000000000000002,
        ];

        let CONV_1_OUTPUT = [
            53.93899999999999,
            54.533,
            53.42700000000001,
            53.651999999999994,
            55.18300000000001,
            54.489000000000004,
            52.412000000000006,
            54.547000000000004,
            54.912000000000006,
        ];

        let LINEAR_OUTPUT = [243.82110000000003];

        let LOSS = [58477.444405210015];

        let LINEAR_WEIGHTS_GRADIENTS = [
            26087.1766258,
            26374.460092600002,
            25839.551819400007,
            25948.3713144,
            26688.827522600008,
            26353.179835800005,
            25348.654986400004,
            26381.231083400005,
            26557.760486400006,
        ];

        let LINEAR_UPDATED_WEIGHTS = [
            -260.771766258,
            -263.54460092600004,
            -258.0955181940001,
            -259.083713144,
            -266.3882752260001,
            -262.93179835800004,
            -252.78654986400005,
            -263.01231083400006,
            -264.67760486400005,
        ];

        let GRADIENTS_FROM_LINEAR_TO_CONV_1 = [
            48.36422000000001,
            96.72844000000002,
            145.09266000000002,
            193.45688000000004,
            241.82110000000003,
            290.18532000000005,
            338.54954000000004,
            386.9137600000001,
            435.27798000000007,
        ];

        let CONV_1_UPDATED_WEIGHTS = [
            -216.76473697200004,
            -219.33472558800005,
            -215.67369597000004,
            -214.85136239400003,
            -220.053503932,
            -218.86559265400007,
            -209.81195250600007,
            -217.30826477000005,
            -218.04638880200008,
        ];

        let GRADIENTS_FROM_CONV_1_TO_CONV_0 = [
            48.36422000000001,
            120.91055000000003,
            203.12972400000004,
            91.89201800000002,
            29.018532000000008,
            217.63899000000004,
            435.27798000000007,
            643.244126,
            386.9137600000001,
            130.58339400000003,
            444.95082400000007,
            914.0837580000002,
            1305.8339400000004,
            875.3923820000002,
            377.2409160000001,
            207.96614600000004,
            677.0990800000001,
            1146.2320140000002,
            1015.6486200000002,
            507.8243100000001,
            67.70990800000001,
            246.65752200000003,
            619.0620160000001,
            604.5527500000001,
            435.27798000000007,
        ];

        let CONV_0_UPDATED_WEIGHTS = [
            -143.09194615400003,
            -147.60589274200004,
            -146.68370100200005,
            -136.883545168,
            -135.508436458,
            -139.321101856,
            -150.684986858,
            -150.33022135400003,
            -143.82694046200004,
            -133.96234628000002,
            -135.86462498800003,
            -145.00048431000002,
            -152.83050969200005,
            -149.04373310200003,
            -144.53632963400003,
            -130.15608216600003,
            -141.902044506,
            -147.46705953000006,
            -150.64145906000002,
            -146.89977513000005,
            -126.53687364800004,
            -135.708294622,
            -142.09393652400001,
            -142.39222982600003,
            -142.84045221000002,
        ];
        //Pytorch matches END

        let bias: Vec<Vec<f64>> = vec![];
        let p_s = (0 as u16, 1 as u16);
        let mut conv_layer_0 = layer_factory::<f64>(
            Layers::Conv2D,
            input_layer.len(),
            conv_0_weights.len(),
            input_layer[0].len(),
            Some(p_s),
            0.0,
        );

        conv_layer_0.set_first_layer_flag();

        let (mut flat_input_layer, mut flat_conv_0_kernel, mut flat_conv_0_bias) =
            conv_layer_0.flatten(input_layer, conv_0_weights, bias);
        let forward_conv_0 = (
            flat_input_layer.as_slice(),
            flat_conv_0_kernel.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_conv_0: Vec<f64> = conv_layer_0.forward(forward_conv_0, None);
        for i in 0..CONV_0_OUTPUT.len() {
            assert!(
                (CONV_0_OUTPUT[i] - z_conv_0[i]).abs() < f64::EPSILON,
                "CONV_0_OUTPUT {} truth {} prediction {}",
                i,
                CONV_0_OUTPUT[i],
                z_conv_0[i]
            );
        }

        let conv_layer_1 = layer_factory::<f64>(
            Layers::Conv2D,
            5, //configuration value
            conv_1_weights.len(),
            5, //configuration value
            Some(p_s),
            0.0,
        );
        let mut flat_conv_1_kernel = conv_layer_1.flatten_kernel(conv_1_weights);
        let forward_conv_1 = (
            z_conv_0.as_slice(),
            flat_conv_1_kernel.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_conv_1: Vec<f64> = conv_layer_1.forward(forward_conv_1, None);
        for i in 0..CONV_1_OUTPUT.len() {
            assert!(
                (CONV_1_OUTPUT[i] - z_conv_1[i]).abs() < f64::EPSILON,
                "CONV_1_OUTPUT {} truth {} prediction {}",
                i,
                CONV_1_OUTPUT[i],
                z_conv_1[i]
            );
        }

        let linear_layer = layer_factory::<f64>(
            Layers::Linear,
            1, //configuration value. Vector already flattened from previous layer
            linear_weights.len(),
            1, //configuration value. Vector already flattened from previous layer
            Some(p_s),
            0.0,
        );

        let mut flat_linear_weights = linear_layer.flatten_kernel(linear_weights);

        let forward_linear = (
            z_conv_1.as_slice(),
            flat_linear_weights.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_linear: Vec<f64> = linear_layer.forward(forward_linear, None);
        assert!(
            LINEAR_OUTPUT[0] - z_linear[0] < f64::EPSILON,
            "LINEAR_OUTPUT truth {} prediction {}",
            LINEAR_OUTPUT[0],
            z_linear[0]
        );

        let (flat_loss, squared) =
            loss_function_factory(LossFunctions::MeanSquares, vec![vec![2.0]], 1.0);
        let mut loss = squared.forward(&flat_loss, &z_linear);
        assert!(
            LOSS[0] - loss[0] < f64::EPSILON,
            "LOSS truth {} prediction {}",
            LOSS[0],
            loss[0]
        );

        //BACKPASS
        let mut from_loss_to_linear_grads = squared.backward(z_linear[0], z_conv_1.as_mut_slice());

        for i in 0..LINEAR_WEIGHTS_GRADIENTS.len() {
            assert!(
                (LINEAR_WEIGHTS_GRADIENTS[i] - from_loss_to_linear_grads[i]).abs() < f64::EPSILON,
                "LINEAR_WEIGHTS_GRADIENTS {} truth {} prediction {}",
                i,
                LINEAR_WEIGHTS_GRADIENTS[i],
                from_loss_to_linear_grads[i]
            );
        }

        let linear_backward = (
            from_loss_to_linear_grads.as_mut_slice(),
            flat_linear_weights.as_mut_slice(),
            z_conv_1.as_mut_slice(),
        );

        let (mut from_linear_to_conv_1_grads, dummy_bias) =
            linear_layer.backward(linear_backward, 0.01, z_linear[0]);

        for i in 0..LINEAR_UPDATED_WEIGHTS.len() {
            assert!(
                (LINEAR_UPDATED_WEIGHTS[i] - flat_linear_weights[i]).abs() < f64::EPSILON,
                "LINEAR_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                LINEAR_UPDATED_WEIGHTS[i],
                flat_linear_weights[i]
            );
        }

        for i in 0..GRADIENTS_FROM_LINEAR_TO_CONV_1.len() {
            assert!(
                (GRADIENTS_FROM_LINEAR_TO_CONV_1[i] - from_linear_to_conv_1_grads[i]).abs()
                    < f64::EPSILON,
                "GRADIENTS_FROM_LINEAR_TO_CONV_1 {} truth {} prediction {}",
                i,
                GRADIENTS_FROM_LINEAR_TO_CONV_1[i],
                from_linear_to_conv_1_grads[i]
            );
        }

        let conv_1_backward = (
            z_conv_0.as_mut_slice(),
            from_linear_to_conv_1_grads.as_mut_slice(),
            flat_conv_1_kernel.as_mut_slice(),
        );

        let (flat_conv_1_kernel, mut from_conv_1_to_conv_0_grads) =
            conv_layer_1.backward(conv_1_backward, 0.01, 0.0);

        for i in 0..CONV_1_UPDATED_WEIGHTS.len() {
            assert!(
                (CONV_1_UPDATED_WEIGHTS[i] - flat_conv_1_kernel[i]).abs() < f64::EPSILON,
                "CONV_1_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                CONV_1_UPDATED_WEIGHTS[i],
                flat_conv_1_kernel[i]
            );
        }

        for i in 0..GRADIENTS_FROM_CONV_1_TO_CONV_0.len() {
            assert!(
                (GRADIENTS_FROM_CONV_1_TO_CONV_0[i] - from_conv_1_to_conv_0_grads[i]).abs()
                    < f64::EPSILON,
                "GRADIENTS_FROM_CONV_1_TO_CONV_0 {} truth {} prediction {}",
                i,
                GRADIENTS_FROM_CONV_1_TO_CONV_0[i],
                from_conv_1_to_conv_0_grads[i]
            );
        }

        let conv_0_backward = (
            flat_input_layer.as_mut_slice(),
            from_conv_1_to_conv_0_grads.as_mut_slice(),
            flat_conv_0_kernel.as_mut_slice(),
        );

        let (flat_conv_0_kernel, _not_needed) = conv_layer_0.backward(conv_0_backward, 0.01, 0.0);

        for i in 0..CONV_0_UPDATED_WEIGHTS.len() {
            assert!(
                (CONV_0_UPDATED_WEIGHTS[i] - flat_conv_0_kernel[i]).abs() < f64::EPSILON,
                "GRADIENTS_FROM_CONV_1_TO_CONV_0 {} truth {} prediction {}",
                i,
                CONV_0_UPDATED_WEIGHTS[i],
                flat_conv_0_kernel[i]
            );
        }
    }

    #[test]
    fn test_conv2d_conv2d_linear_7_5_3_32() {
        let mut input_layer: Vec<Vec<f32>> = vec![
            vec![1.0, 0.5, 1.2, 0.8, 1.5, 0.9, 1.3, 0.7, 1.1],
            vec![0.6, 1.4, 0.8, 1.7, 1.0, 1.6, 0.9, 1.2, 0.5],
            vec![1.3, 0.7, 1.8, 1.1, 1.4, 0.6, 1.9, 1.0, 1.5],
            vec![0.9, 1.2, 0.8, 1.6, 1.3, 1.1, 0.7, 1.4, 0.9],
            vec![1.1, 0.8, 1.5, 1.0, 1.7, 1.2, 1.4, 0.6, 1.3],
            vec![0.7, 1.3, 0.9, 1.4, 1.1, 1.8, 1.0, 1.5, 0.8],
            vec![1.2, 0.6, 1.4, 0.9, 1.3, 1.0, 1.6, 1.1, 1.7],
            vec![0.8, 1.1, 0.7, 1.5, 1.2, 1.4, 0.9, 1.3, 1.0],
            vec![1.0, 1.5, 0.8, 1.2, 0.9, 1.3, 1.1, 0.7, 1.4],
        ];

        let conv_0_weights: Vec<Vec<f32>> = vec![
            vec![0.1, 0.2, 0.3, 0.2, 0.1],
            vec![0.2, 0.4, 0.6, 0.4, 0.2],
            vec![0.3, 0.6, 0.9, 0.6, 0.3],
            vec![0.2, 0.4, 0.6, 0.4, 0.2],
            vec![0.1, 0.2, 0.3, 0.2, 0.1],
        ];

        let conv_1_weights: Vec<Vec<f32>> = vec![
            vec![1.0, 0.5, 0.2],
            vec![0.5, 1.0, 0.5],
            vec![0.2, 0.5, 1.0],
        ];

        let linear_weights: Vec<Vec<f32>> = vec![
            vec![0.1],
            vec![0.2],
            vec![0.3],
            vec![0.4],
            vec![0.5],
            vec![0.6],
            vec![0.7],
            vec![0.8],
            vec![0.9],
        ];

        let linear_bias: Vec<Vec<f32>> = vec![vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];

        //Pytorch matches
        let CONV_0_OUTPUT: [f32; 25] = [
            9.55, 9.960001, 10.09, 9.63, 9.41, 9.59, 10.17, 10.27, 9.87, 9.44, 9.430001, 10.01,
            10.4, 10.11, 9.78, 9.040001, 9.82, 10.280001, 10.339999, 9.97, 8.81, 9.450001, 9.93,
            10.1, 10.000001,
        ];

        let CONV_1_OUTPUT: [f32; 9] = [
            53.939003, 54.533, 53.427002, 53.652, 55.183, 54.489, 52.412003, 54.547005, 54.912003,
        ];

        let LINEAR_OUTPUT: [f32; 1] = [243.821121];

        let LOSS: [f32; 1] = [58477.4531];

        let LINEAR_WEIGHTS_GRADIENTS: [f32; 9] = [
            26087.18, 26374.463, 25839.555, 25948.373, 26688.83, 26353.182, 25348.658, 26381.236,
            26557.764,
        ];

        let LINEAR_UPDATED_WEIGHTS: [f32; 9] = [
            -260.7718, -263.54462, -258.09555, -259.0837, -266.3883, -262.93182, -252.78658,
            -263.01236, -264.67764,
        ];

        let GRADIENTS_FROM_LINEAR_TO_CONV_1: [f32; 9] = [
            48.364223, 96.72845, 145.09268, 193.4569, 241.82112, 290.18536, 338.54956, 386.9138,
            435.278,
        ];

        let CONV_1_UPDATED_WEIGHTS: [f32; 9] = [
            -216.76476, -219.33475, -215.6737, -214.85138, -220.05351, -218.86562, -209.81198,
            -217.30827, -218.0464,
        ];

        let GRADIENTS_FROM_CONV_1_TO_CONV_0: [f32; 25] = [
            48.364223, 120.91056, 203.12976, 91.89203, 29.018538, 217.639, 435.278, 643.2442,
            386.91382, 130.58342, 444.95087, 914.08386, 1305.8341, 875.39246, 377.24097, 207.96616,
            677.0991, 1146.2322, 1015.64874, 507.82437, 67.709915, 246.65753, 619.0621, 604.5528,
            435.278,
        ];

        let CONV_0_UPDATED_WEIGHTS: [f32; 25] = [
            -143.09196, -147.6059, -146.6837, -136.88354, -135.50844, -139.3211, -150.68501,
            -150.33023, -143.82695, -133.96236, -135.86462, -145.00049, -152.83052, -149.04375,
            -144.53635, -130.1561, -141.90205, -147.46707, -150.64146, -146.89978, -126.53689,
            -135.7083, -142.09395, -142.39224, -142.84047,
        ];
        //Pytorch matches END

        let bias: Vec<Vec<f32>> = vec![];
        let p_s = (0 as u16, 1 as u16);
        let mut conv_layer_0 = layer_factory::<f32>(
            Layers::Conv2D,
            input_layer.len(),
            conv_0_weights.len(),
            input_layer[0].len(),
            Some(p_s),
            0.0,
        );

        conv_layer_0.set_first_layer_flag();

        let (mut flat_input_layer, mut flat_conv_0_kernel, mut flat_conv_0_bias) =
            conv_layer_0.flatten(input_layer, conv_0_weights, bias);
        let forward_conv_0 = (
            flat_input_layer.as_slice(),
            flat_conv_0_kernel.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_conv_0: Vec<f32> = conv_layer_0.forward(forward_conv_0, None);
        for i in 0..CONV_0_OUTPUT.len() {
            assert!(
                (CONV_0_OUTPUT[i] - z_conv_0[i]).abs() < f32::EPSILON,
                "CONV_0_OUTPUT {} truth {} prediction {}",
                i,
                CONV_0_OUTPUT[i],
                z_conv_0[i]
            );
        }

        let conv_layer_1 = layer_factory::<f32>(
            Layers::Conv2D,
            5, //configuration value
            conv_1_weights.len(),
            5, //configuration value
            Some(p_s),
            0.0,
        );
        let mut flat_conv_1_kernel = conv_layer_1.flatten_kernel(conv_1_weights);
        let forward_conv_1 = (
            z_conv_0.as_slice(),
            flat_conv_1_kernel.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_conv_1: Vec<f32> = conv_layer_1.forward(forward_conv_1, None);

        for i in 0..CONV_1_OUTPUT.len() {
            assert!(
                (CONV_1_OUTPUT[i] - z_conv_1[i]).abs() < f32::EPSILON,
                "CONV_1_OUTPUT {} truth {} prediction {}",
                i,
                CONV_1_OUTPUT[i],
                z_conv_1[i]
            );
        }

        let linear_layer = layer_factory::<f32>(
            Layers::Linear,
            1, //configuration value. Vector already flattened from previous layer
            linear_weights.len(),
            1, //configuration value. Vector already flattened from previous layer
            Some(p_s),
            0.0,
        );

        let mut flat_linear_weights = linear_layer.flatten_kernel(linear_weights);

        let forward_linear = (
            z_conv_1.as_slice(),
            flat_linear_weights.as_mut_slice(),
            flat_conv_0_bias.as_slice(),
        );

        let mut z_linear: Vec<f32> = linear_layer.forward(forward_linear, None);

        assert!(
            LINEAR_OUTPUT[0] - z_linear[0] < f32::EPSILON,
            "LINEAR_OUTPUT truth {} prediction {}",
            LINEAR_OUTPUT[0],
            z_linear[0]
        );

        let (flat_loss, squared) =
            loss_function_factory(LossFunctions::MeanSquares, vec![vec![2.0]], 1.0);
        let mut loss = squared.forward(&flat_loss, &z_linear);
        assert!(
            LOSS[0] - loss[0] < f32::EPSILON,
            "LOSS truth {} prediction {}",
            LOSS[0],
            loss[0]
        );

        //BACKPASS
        let mut from_loss_to_linear_grads = squared.backward(z_linear[0], z_conv_1.as_mut_slice());

        for i in 0..LINEAR_WEIGHTS_GRADIENTS.len() {
            assert!(
                (LINEAR_WEIGHTS_GRADIENTS[i] - from_loss_to_linear_grads[i]).abs() < f32::EPSILON,
                "LINEAR_WEIGHTS_GRADIENTS {} truth {} prediction {}",
                i,
                LINEAR_WEIGHTS_GRADIENTS[i],
                from_loss_to_linear_grads[i]
            );
        }

        let linear_backward = (
            from_loss_to_linear_grads.as_mut_slice(),
            flat_linear_weights.as_mut_slice(),
            z_conv_1.as_mut_slice(),
        );

        let (mut from_linear_to_conv_1_grads, dummy_bias) =
            linear_layer.backward(linear_backward, 0.01, z_linear[0]);
        for i in 0..LINEAR_UPDATED_WEIGHTS.len() {
            assert!(
                (LINEAR_UPDATED_WEIGHTS[i] - flat_linear_weights[i]).abs() < f32::EPSILON,
                "LINEAR_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                LINEAR_UPDATED_WEIGHTS[i],
                flat_linear_weights[i]
            );
        }

        for i in 0..GRADIENTS_FROM_LINEAR_TO_CONV_1.len() {
            assert!(
                (GRADIENTS_FROM_LINEAR_TO_CONV_1[i] - from_linear_to_conv_1_grads[i]).abs()
                    < f32::EPSILON,
                "LINEAR_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                GRADIENTS_FROM_LINEAR_TO_CONV_1[i],
                from_linear_to_conv_1_grads[i]
            );
        }

        let conv_1_backward = (
            z_conv_0.as_mut_slice(),
            from_linear_to_conv_1_grads.as_mut_slice(),
            flat_conv_1_kernel.as_mut_slice(),
        );

        let (flat_conv_1_kernel, mut from_conv_1_to_conv_0_grads) =
            conv_layer_1.backward(conv_1_backward, 0.01, 0.0);

        for i in 0..CONV_1_UPDATED_WEIGHTS.len() {
            assert!(
                (CONV_1_UPDATED_WEIGHTS[i] - flat_conv_1_kernel[i]).abs() < f32::EPSILON,
                "CONV_1_UPDATED_WEIGHTS {} truth {} prediction {}",
                i,
                CONV_1_UPDATED_WEIGHTS[i],
                flat_conv_1_kernel[i]
            );
        }

        for i in 0..GRADIENTS_FROM_CONV_1_TO_CONV_0.len() {
            assert!(
                (GRADIENTS_FROM_CONV_1_TO_CONV_0[i] - from_conv_1_to_conv_0_grads[i]).abs()
                    < f32::EPSILON,
                "GRADIENTS_FROM_CONV_1_TO_CONV_0 {} truth {} prediction {}",
                i,
                GRADIENTS_FROM_CONV_1_TO_CONV_0[i],
                from_conv_1_to_conv_0_grads[i]
            );
        }

        let conv_0_backward = (
            flat_input_layer.as_mut_slice(),
            from_conv_1_to_conv_0_grads.as_mut_slice(),
            flat_conv_0_kernel.as_mut_slice(),
        );

        let (flat_conv_0_kernel, _not_needed) = conv_layer_0.backward(conv_0_backward, 0.01, 0.0);

        for i in 0..CONV_0_UPDATED_WEIGHTS.len() {
            assert!(
                (CONV_0_UPDATED_WEIGHTS[i] - flat_conv_0_kernel[i]).abs() < f32::EPSILON,
                "GRADIENTS_FROM_CONV_1_TO_CONV_0 {} truth {} prediction {}",
                i,
                CONV_0_UPDATED_WEIGHTS[i],
                flat_conv_0_kernel[i]
            );
        }
    }

}
