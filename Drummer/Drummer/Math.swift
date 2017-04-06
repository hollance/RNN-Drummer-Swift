import Foundation
import Accelerate

/* Simple wrapper functions around Accelerate. */

enum Math {
  /* Returns a new random number in the range [0, upperBound) (exclusive). */
  static func random(_ upperBound: Int) -> Int {
    return Int(arc4random_uniform(UInt32(upperBound)))
  }

  /* Returns a new random number in the range [0, 1) (exclusive). */
  static func random() -> Float {
    return Float(arc4random()) / Float(0x100000000)
  }

  /*
    Fills up the given array with uniformly random values between -magnitude
    and +magnitude.
  */
  static func uniformRandom(_ x: UnsafeMutablePointer<Float>, _ count: Int, _ magnitude: Float) {
    for i in 0..<count {
      x[i] = (random()*2 - 1) * magnitude
    }
  }

  /*
    Matrix multiplication: C = A * B

    M: Number of rows in matrices A and C.
    N: Number of columns in matrices B and C. 
    K: Number of columns in matrix A; number of rows in matrix B.
  */
  static func matmul(_ A: UnsafePointer<Float>, _ B: UnsafePointer<Float>, _ C: UnsafeMutablePointer<Float>, _ M: Int, _ N: Int, _ K: Int) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, Int32(M), Int32(N),
                Int32(K), 1, A, Int32(K), B, Int32(N), 0, C, Int32(N))
  }

  static func add(_ x: UnsafeMutablePointer<Float>, _ y: Float, _ count: Int) {
    var y = y
    vDSP_vsadd(x, 1, &y, x, 1, vDSP_Length(count))
  }

  static func add(_ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<Float>, _ count: Int) {
    cblas_saxpy(Int32(count), 1, x, 1, y, 1)
  }

  static func divide(_ x: UnsafeMutablePointer<Float>, _ y: Float, _ count: Int) {
    var y = y
    vDSP_vsdiv(x, 1, &y, x, 1, vDSP_Length(count))
  }

  static func exp(_ x: UnsafeMutablePointer<Float>, _ count: Int) {
    var cnt = Int32(count)
    vvexpf(x, x, &cnt)
  }

  static func max(_ x: UnsafeMutablePointer<Float>, _ count: Int) -> Float {
    var y: Float = 0
    vDSP_maxv(x, 1, &y, vDSP_Length(count))
    return y
  }

  /* Multiply two vectors element-wise: z[i] = x[i] * y[i] */
  static func multiply(_ x: UnsafePointer<Float>, _ y: UnsafePointer<Float>, _ z: UnsafeMutablePointer<Float>, _ count: Int) {
    vDSP_vmul(x, 1, y, 1, z, 1, vDSP_Length(count))
  }

  static func negate(_ x: UnsafeMutablePointer<Float>, _ count: Int) {
    vDSP_vneg(x, 1, x, 1, vDSP_Length(count))
  }

  static func reciprocal(_ x: UnsafeMutablePointer<Float>, _ count: Int) {
    var cnt = Int32(count)
    vvrecf(x, x, &cnt)
  }

  static func sum(_ x: UnsafeMutablePointer<Float>, _ count: Int) -> Float {
    var y: Float = 0
    vDSP_sve(x, 1, &y, vDSP_Length(count))
    return y
  }

  static func tanh(_ x: UnsafeMutablePointer<Float>, _ count: Int) {
    var cnt = Int32(count)
    vvtanhf(x, x, &cnt)
  }

  static func tanh(_ x: UnsafePointer<Float>, _ y: UnsafeMutablePointer<Float>, _ count: Int) {
    var cnt = Int32(count)
    vvtanhf(y, x, &cnt)
  }

  /* Logistic sigmoid: 1 / (1 + np.exp(-x)) */
  static func sigmoid(_ x: UnsafeMutablePointer<Float>, _ count: Int) {
    Math.negate(x, count)
    Math.exp(x, count)
    Math.add(x, 1, count)
    Math.reciprocal(x, count)
  }

  /* The softmax turns the array into a probability distribution. */
  static func softmax(_ x: UnsafeMutablePointer<Float>, _ count: Int) {
    // Find the maximum value in the input array.
    let max = Math.max(x, count)

    // Subtract the maximum from all the elements in the array.
    // Now the highest value in the array is 0.
    Math.add(x, -max, count)

    // Exponentiate all the elements in the array.
    Math.exp(x, count)

    // Compute the sum of all exponentiated values.
    let sum = Math.sum(x, count)

    // Divide each element by the sum. This normalizes the array
    // contents so that they all add up to 1.
    Math.divide(x, sum, count)
  }

  /* Returns a number between 0 and count, using the probabilities in x. */
  static func randomlySample(_ x: UnsafeMutablePointer<Float>, _ count: Int) -> Int {
    // Compute the cumulative sum of the probabilities.
    var cumsum = [Float](repeating: 0, count: count)
    var sum: Float = 0
    for i in 0..<count {
      sum += x[i]
      cumsum[i] = sum
    }

    // Normalize so that the last element is exactly 1.0.
    Math.divide(&cumsum, cumsum.last!, count)

    // Get a new random number between 0 and 1 (exclusive).
    let sample = random()

    // Find the index of where sample would go in the array.
    for i in stride(from: count - 2, through: 0, by: -1) {
      if cumsum[i] <= sample {
        return i + 1
      }
    }
    return 0
  }
}
