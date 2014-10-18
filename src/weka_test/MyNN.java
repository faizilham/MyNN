package weka_test;

import java.util.ArrayList;
import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class MyNN extends Classifier{
	private static final long serialVersionUID = 5510340142122736634L;
	
	public enum ANNType {PTR, BATCH_GRADIENT_DESCENT, DELTA_RULE, BACKPROPAGATION}
	
	private static class Layer{
		public double[] output, error, bias;
		public ActivationFunction function;
		public double[][] weight;
		public double[][] prevWeight;
		public int size;
		
		public Layer(int inputSize){ // untuk input layer
			output = new double[inputSize]; error = null; weight = null; prevWeight = null; function = null; bias = null;
			size = inputSize;
		}
		
		public Layer(int size, int inputSize, ActivationFunction func){ // untuk hidden & output layer
			output = new double[size]; error = new double[size]; 
			weight = new double[size][inputSize];
			prevWeight = new double[size][inputSize];
			function = func; bias = new double[size];
			this.size = size;
		}
	}
	
//	ArrayList<double[]> values = new ArrayList<>();
//	ArrayList<double[]> errors = new ArrayList<>();
//	ArrayList<double[]> bias = new ArrayList<>();
//	ArrayList<ActivationFunction> functions = new ArrayList<>();
//	ArrayList<double[][]> weights = new ArrayList<>();
	ClassificationFunction classFunc;
	
	ArrayList<Layer> layers = new ArrayList<Layer>();
	
	private static final double MAX_RANDOM = 1;
	private static final double MAX_RANDOM_RANGE = MAX_RANDOM * 2;
	private ANNType type;
	public int maxEpoch = 500;
	public final double learnRate = 0.3;
	public double errorMin = 0.01;
	public double convergenceError = 0.01;
	public double momentum = 0.7;
	
	
	public MyNN(int inputSize, ClassificationFunction classFunc, ANNType type){
		this.classFunc = classFunc;
		this.type = type;
		
		layers.add(new Layer(inputSize));
	}
	
	public void addLayer(int size, ActivationFunction func){
		int prevSize = layers.get(layers.size()-1).size;
		
		layers.add(new Layer(size,prevSize,func));
	}
	
	private void PTR(Instances data){
		double error; int epoch = 0;
		
		Layer current, prev;
		do{
					
			for (int i = 0; i < data.numInstances(); ++i){
				// classify
				double y = data.instance(i).classValue();
				double o = classifyInstance(data.instance(i));
				double e = y - o;
				
				// update weight
				for (int w = 1; w < layers.size(); ++w){
					current = layers.get(w); prev = layers.get(w-1);
					
					for (int r = 0; r < current.size; ++r){
						for (int c = 0; c < prev.size; ++c){
							double update = learnRate * e * prev.output[c];
							double dw = current.weight[r][c] - current.prevWeight[r][c];
							current.prevWeight[r][c] = current.weight[r][c];
							
							current.weight[r][c] += update + (momentum * dw);
						}
					}
				}
				
				// update bias
				for (int w = 1; w < layers.size(); ++w){
					current = layers.get(w);
					
					for (int j = 0; j < current.size; j++){
						double update = learnRate * e; 
						current.bias[j] += update;
					}
				}
			}
			epoch += 1;

			// recalculate error
			
			error = 0;
			
			for (int i = 0; i < data.numInstances(); ++i){
				// classify
				double y = data.instance(i).classValue();
				double o = classifyInstance(data.instance(i));
				double e = y - o;
				error += Math.abs(e);
			}
			
			error = error / data.numInstances();
			
			System.out.printf("epoch %d: %.2f\n", epoch, error);
		}while(error > errorMin && epoch < maxEpoch);
	}
	
	private void batchGradientDescent(Instances data){
		double error; int epoch = 0;
		Layer current, prev;
		
		do {
			double[] e = new double[data.numInstances()];
			
			// classify
			for (int i = 0; i < data.numInstances(); i++) {
				double y = data.instance(i).classValue();
				double o = classifyInstance(data.instance(i));
				e[i] = y - o;
			}
			
			double E = 0;
			
			// update weight
			for (int w = 1; w < layers.size(); ++w){
				current = layers.get(w); prev = layers.get(w-1);

				for (int i = 0; i < data.numInstances(); i++) {
					for (int j = 0; j < prev.size; j++) {
						E += e[i] * prev.output[j];
					}
				}
				
				for (int r = 0; r < current.size; ++r){
					for (int c = 0; c < prev.size; ++c){
						double update = learnRate * E / data.numInstances();
						double dw = current.weight[r][c] - current.prevWeight[r][c];
						current.prevWeight[r][c] = current.weight[r][c];
						
						current.weight[r][c] += update + (momentum * dw);
					}
				}
			}
			
			// update bias
			for (int w = 1; w < layers.size(); ++w){
				current = layers.get(w);
				
				for (int j = 0; j < current.size; j++){
					double update = learnRate * E / data.numInstances(); 
					current.bias[j] += update;
				}
			}
			
			epoch += 1;
			
			// recalculate error
			error = 0;
			for (int i = 0; i < data.numInstances(); ++i){
				error += e[i] * e[i];
			}
			
			error /= 2.0;
			
			System.out.printf("epoch %d: %.2f\n", epoch, error);
		} while(error > errorMin && epoch < maxEpoch);
	}
	
	private void deltaRule(Instances data){
		double error; int epoch = 0;
		
		Layer current, prev;
		do{
					
			for (int i = 0; i < data.numInstances(); ++i){
				// classify
				double y = data.instance(i).classValue();
				double o = classifyInstance(data.instance(i));
				double e = y - o;
				
				// update weight
				for (int w = 1; w < layers.size(); ++w){
					current = layers.get(w); prev = layers.get(w-1);
					
					for (int r = 0; r < current.size; ++r){
						for (int c = 0; c < prev.size; ++c){
							double update = learnRate * e * prev.output[c];
							double dw = current.weight[r][c] - current.prevWeight[r][c];
							current.prevWeight[r][c] = current.weight[r][c];
							
							current.weight[r][c] += update + (momentum * dw);
						}
					}
				}
				
				// update bias
				for (int w = 1; w < layers.size(); ++w){
					current = layers.get(w);
					
					for (int j = 0; j < current.size; j++){
						double update = learnRate * e; 
						current.bias[j] += update;
					}
				}
			}
			epoch += 1;

			// recalculate error
			error = 0;
			for (int i = 0; i < data.numInstances(); ++i){
				double y = data.instance(i).classValue();
				double o = classifyInstance(data.instance(i));
				double e = y - o;
				error += e * e;
			}
			
			error /= 2.0;
			
			System.out.printf("epoch %d: %.2f\n", epoch, error);
		}while(error > errorMin && epoch < maxEpoch);
	}
	
	private void backpropagationLearn(Instances data){
		double error; int epoch = 0;
		
		//double learnRate = this.learnRate;
		
		//double[] err, value, nextErr, b; double[][] weight;
		Layer current, prev, next;
		do{
					
			for (int i = 0; i < data.numInstances(); ++i){
				// classify
				double y = data.instance(i).classValue();
				double o = classifyInstance(data.instance(i));
				double e = y - o;
				
				// backpropagate
				// count output layer error
								
				current = layers.get(layers.size()-1);
				
				for (int j = 0; j < current.size; ++j){
					current.error[j] = current.output[j]*(1-current.output[j])*e; // TODO edit derivation error
				}
				
				// count hidden error
				for (int h = layers.size()-2; h > 0; --h){
					
					current = layers.get(h); next = layers.get(h+1);
					
					for (int j = 0; j < current.size; j++){
						double sum = 0;
						// sum weight * output error
						for (int k = 0; k < next.size; ++k){
							sum += next.weight[k][j] * next.error[k];
						}
						
						current.error[j] = current.output[j] * (1-current.output[j]) * sum;
					}
				}
				
				// update weight
				
				for (int w = 1; w < layers.size(); ++w){
					current = layers.get(w); prev = layers.get(w-1);
					
					for (int r = 0; r < current.size; ++r){
						for (int c = 0; c < prev.size; ++c){
							double update = learnRate * current.error[r] * prev.output[c];
							double dw = current.weight[r][c] - current.prevWeight[r][c];
							current.prevWeight[r][c] = current.weight[r][c];
							
							current.weight[r][c] += update + (momentum * dw);
						}
					}
				}
				
				// update bias
				for (int w = 1; w < layers.size(); ++w){
					current = layers.get(w);
					
					for (int j = 0; j < current.size; j++){
						double update = learnRate * current.error[j]; 
						current.bias[j] += update;
					}
				}
				
				
				
			}
			epoch += 1;

			// recalculate error
			
			error = 0;
			
			for (int i = 0; i < data.numInstances(); ++i){
				// classify
				double y = data.instance(i).classValue();
				double o = classifyInstance(data.instance(i));
				double e = y - o;
				error += Math.abs(e);
			}
			
			error = error / data.numInstances();
			
			System.out.printf("epoch %d: %.2f\n", epoch, error);
		}while(error > errorMin && epoch < maxEpoch);
	}
	
	private void randomize(){
		Random rand = new Random();
		for (int l = 1; l < layers.size(); ++l){
			double[][] weight = layers.get(l).weight;
			double[][] prevWeight = layers.get(l).prevWeight;
			double[] bias = layers.get(l).bias;
			
			for (int i = 0; i < weight.length; ++i){
				for (int j = 0; j < weight[0].length; ++j){
					weight[i][j] = (rand.nextFloat() * MAX_RANDOM_RANGE) - MAX_RANDOM;
					prevWeight[i][j] = weight[i][j];
				}
			}
			
			for (int i = 0; i < bias.length; ++i){
				bias[i] = (rand.nextFloat() * MAX_RANDOM_RANGE) - MAX_RANDOM;
			}
		}
	}
	

	
	public void print(){
		System.out.print("w --> ");
		for (int w = 1; w < layers.size(); ++w){
			double[][] weight = layers.get(w).weight;
			for (int i = 0; i < weight.length; ++i){
				for (int j = 0; j < weight[0].length; ++j){
					System.out.printf("%.2f ", weight[i][j]);
				}
				//System.out.println();
			}
			
			
		}
		
		System.out.println();
		
		System.out.print("b --> ");
		
		for (int w = 1; w < layers.size(); ++w){
			double[] bias = layers.get(w).bias;
			for (int i = 0; i < bias.length; ++i){
				System.out.printf("%.2f ", bias[i]);
			}

		}
		System.out.println();
	}

	@Override
	public void buildClassifier(Instances data) throws Exception {
		
		assert layers.size() > 1;
		
		randomize();
		print();
		// start learning
		switch(type){
		case PTR:
			PTR(data);
			break;
		case BATCH_GRADIENT_DESCENT:
			batchGradientDescent(data);
			break;
		case DELTA_RULE:
			deltaRule(data);
			break;
		case BACKPROPAGATION:
			backpropagationLearn(data);
			break;
		
		}
		
	}
	
	private void copyInput(Instance instance){
		double[] inputLayer = layers.get(0).output;
		
		
		for (int i = 0; i < inputLayer.length; ++i){
			inputLayer[i] = instance.value(i);
		}
	}
	
	@Override
	public double classifyInstance(Instance instance){
		copyInput(instance);
		
		for (int v = 1; v < layers.size(); ++v){			
			Layer current = layers.get(v), prev = layers.get(v-1);
			
			for (int i = 0; i < current.size; ++i){
				double value = current.bias[i];
				for (int j = 0; j < prev.size; ++j){
					value += current.weight[i][j] * prev.output[j];
				}
				
				if (type.equals(ANNType.DELTA_RULE)) {
					current.output[i] = value;
				} else {
					current.output[i] = current.function.activate(value);
				}
			}
		}
		
		double[] outputLayer = layers.get(layers.size()-1).output;
		
		return classFunc.classify(outputLayer);
	}
	
	/** Activation Functions **/
	
	public static interface ActivationFunction{
		public double activate(double input);
	}
	
	public static class HardLimFunction implements ActivationFunction{
		@Override
		public double activate(double input) {
			if (input > 0)
				return 1;
			else
				return -1;
		}
	}
	
	public static class StepFunction implements ActivationFunction{
		@Override
		public double activate(double input) {
			if (input > 0)
				return 1;
			else
				return 0;
		}
	}
	
	public static class SigmoidFunction implements ActivationFunction{
		@Override
		public double activate(double input) {
			return 1 / (1 + Math.exp(-input));
		}
	}
	
	/** Classification Functions **/
	
	public static interface ClassificationFunction{
		public double classify(double[] outputLayer);
	}
	
	public static class IndexFunction implements ClassificationFunction{
		int index;
		public IndexFunction(int index){ this.index = index;}
		@Override
		public double classify(double[] outputLayer) {
			return (outputLayer[index] > 0.5) ? 1 : 0;
			//return outputLayer[index];
		}
		
	}
}
