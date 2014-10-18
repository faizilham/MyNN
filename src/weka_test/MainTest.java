package weka_test;

import java.io.FileReader;

import weka.core.Instance;
import weka.core.Instances;
import weka_test.MyNN.HardLimFunction;
import weka_test.MyNN.IndexFunction;

public class MainTest {
	public static void main(String[] args) throws Exception{
		FileReader freader = new FileReader("data/and.arff");
		Instances data = new Instances(freader);
		data.setClassIndex(data.numAttributes()-1);
		freader.close();

		//Test PTR
		//MyNN nn = new MyNN(2, new MyNN.IndexFunction(0), MyNN.ANNType.PTR);
		//nn.addLayer(1, new MyNN.HardLimFunction());

		// Test Batch
		MyNN nn = new MyNN(2, new MyNN.IndexFunction(0), MyNN.ANNType.BATCH_GRADIENT_DESCENT);
		nn.addLayer(1, new MyNN.SigmoidFunction());
		
		// Test Delta
		//MyNN nn = new MyNN(2, new MyNN.IndexFunction(0), MyNN.ANNType.DELTA_RULE);
		//nn.addLayer(1, new MyNN.HardLimFunction());
		
		// Backpropagation
		//MyNN nn = new MyNN(2, new MyNN.IndexFunction(0), MyNN.ANNType.BACKPROPAGATION);
		//nn.addLayer(2, new MyNN.SigmoidFunction());
		//nn.addLayer(1, new MyNN.SigmoidFunction());
		
		nn.buildClassifier(data);
		
		for (int i = 0; i < data.numInstances(); ++i){
			Instance in = data.instance(i);
			System.out.printf("%s : ", in.toString());
			System.out.println(nn.classifyInstance(in));
		}
	}
}
