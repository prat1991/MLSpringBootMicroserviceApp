package prat.classifier.model;

public class InputFeatureRequest {
	private double sepalLength;
	private double sepalWidth;
	private double petalLength;
	private double petalWidth;

	// Default constructor
	public InputFeatureRequest() {
	}

	// Constructor with parameters
	public InputFeatureRequest(double sepalLength,
 double sepalWidth, 
double petalLength, double petalWidth) {
		this.sepalLength = sepalLength;
		this.sepalWidth = sepalWidth;
		this.petalLength = petalLength;
		this.petalWidth = petalWidth;
	}

	// Getters and setters
	public double getSepalLength() {
		return sepalLength;
	}

	public void setSepalLength(double sepalLength) {
		this.sepalLength = sepalLength;
	}

	public double getSepalWidth() {
		return sepalWidth;
	}

	public void setSepalWidth(double sepalWidth) {
		this.sepalWidth = sepalWidth;
	}

	public double getPetalLength() {
		return petalLength;
	}

	public void setPetalLength(double petalLength) {
		this.petalLength = petalLength;
	}

	public double getPetalWidth() {
		return petalWidth;
	}

	public void setPetalWidth(double petalWidth) {
		this.petalWidth = petalWidth;
	}
}