package cs446.weka.classifiers.trees;

import weka.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Instances;

public class SGD extends Classifier{
     

	private static final long serialVersionUID = -2693678647096322561L;
	public static double [] weights =new double[271];
    double alpha=0.01;
	

	public void	 buildClassifier(Instances data) throws Exception{
		
		// can classifier handle the data?
		getCapabilities().testWithFail(data);
        
		// remove instances with missing class
		//data = new Instances(data);
		data.deleteWithMissingClass();
		//weights=new double [data.numAttributes()];
		
		for(int i=0;i<data.numAttributes()-1;i++){
			weights[i]=0.001;
		}
		weights[270]=0.1;
	}

	public double classifyInstance(Instance instance)
            throws java.lang.Exception{
		double result = 0;
		double[] att = instance.toDoubleArray();
				
				if(dotProd(att,weights)<0.5){
				result=0.0;
				
					//instance.setClassValue(0.0);
				}
				else if(dotProd(att,weights)>=0.5){
				result=1.0;
				
					//instance.setClassValue(1.0);
				}
				return result;
		}

	
	protected void updateClassifier(Instance instance, double result)
		      throws Exception {
		
	double y=instance.classValue();
	double[] att =instance.toDoubleArray();
	int j=instance.numAttributes();
	for(int i=0;i<j;i++){
	weights[i]=weights[i]-alpha*(result-y)*att[i];
	
	
	}
	}
	 double dotProd(double [] attribute, double[] weights)
	       {
	   double result = 0;
        for(int i=0;i<=weights.length-1;i++){
        	result=result+weights[i]*attribute[i];
        }
	    return (result);
	  }
	
	 
	boolean convergence(double [] weightsA,double [] weightsB){
		double sum=0;
		for(int j=0;j<=weightsA.length-1;j++){
			sum=sum+Math.pow((weightsA[j]-weightsB[j]),2);
		}
		sum=Math.sqrt(sum);
		System.out.println(sum);
		if(sum<=0.001){
			return true;		
		}
		else return false;
	}
	

}
