let user_digit;
let result;
let user;
let user_has_drawing = false;
let user_guess;


// loading the Neural network data from the json file
function preload() {
	network = loadJSON("mnist.json");
}


//  Creating the canvas in for the user to draw.
function setup() {
	user_guess = select('.guess');
	var cnv = createCanvas(224,224);
	var x = (windowWidth - width) / 2;
	var y = (windowHeight - height) / 2;
	cnv.position(x, y);
	user_digit = createGraphics(224, 224);
	user_digit.pixelDensity(1);

	// calling the guessing function
	user = guessUserDigit();
}


function draw() {
	background(0);

	// allowing the user to draw on the canvas
	if (mouseIsPressed) {
		user_has_drawing = true;
		user_digit.stroke(255);
		user_digit.strokeWeight(9);
		user_digit.line(mouseX, mouseY, pmouseX, pmouseY);

		// calling the guessing function and getting the guess of the network
		user = guessUserDigit();
		user_guess.html(user);
	}
}


// use "space" button to clear the canvas
function keyPressed() {
	if (key == ' ') {
		user_digit.background(0);
		user_guess.html('_');
	}
}


// the function to guess the user's drawing
function guessUserDigit() {
	let img = user_digit.get();
	if(user_has_drawing == false){
		user_guess.html('_');
		return img;
	}else{
	let inputs = [];
	// loading and resizing the user image so that it matches the training data
	img.resize(28, 28);
	img.loadPixels();

	// converting the RGBA values of each pixel to only one value
	for (let i = 0; i < 784; i++) {
		inputs[i] = img.pixels[i * 4]/255.0;
	}
	// testing the network on these inputs
	result = test(network, [inputs])
	num = math.max(result[0])
	return result[0].indexOf(num);
	}
}


// the activation function used by the neural network
function sigmoid(x) {
	for (let i = 0; i < x.length; i++) {
		for (let j = 0; j < x[i].length; j++) {
			let val = x[i][j];
			x[i][j] = 1/(1+math.exp(-val));
		}
	}
	return x;
}

// the testing function
function test(network, inputs) {
	// calculating the output for the hidden layer
		var Hlayer_output;
		Hlayer_output = math.multiply(inputs, network.weights_ih);
		Hlayer_output = sigmoid(Hlayer_output)
	// calculating the final output
		var output;
		output = math.multiply(Hlayer_output, network.weights_ho);
		output = sigmoid(output);
		return output;
}
