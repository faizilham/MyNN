package weka_test;

import java.util.Random;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class RandomClassifier extends Classifier{

	private static final long serialVersionUID = 3875223964873148463L;
	private int numValues;
	private Random rnd;
	
	@Override
	public void buildClassifier(Instances dataset) throws Exception {
		numValues = dataset.classAttribute().numValues();
		rnd = new Random();
	}
	
	@Override
	public double classifyInstance(Instance instance) {
		return rnd.nextInt(numValues);
	}

}
