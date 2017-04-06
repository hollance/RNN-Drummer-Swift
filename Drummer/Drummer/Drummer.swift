import Foundation
import Accelerate

private let Wx_rows = 427
private let Wx_cols = 800

private let Wy_rows = 201
private let Wy_cols = 226

private let hiddenSize = 200

/* These two look-up tables were exported from the Python training script. */
private let index2note = [36, 37, 38, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53]
private let index2tick = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 98, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 136, 154, 159, 161, 166, 208, 214, 220, 221, 222, 223, 224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 346, 347, 348, 349, 350, 351, 352, 353, 354, 355, 356, 357, 358, 359, 360, 361, 362, 363, 365, 367, 458, 463, 467, 468, 469, 470, 471, 472, 473, 474, 475, 476, 477, 478, 479, 480, 487]

private let noteVectorSize = index2note.count
private let tickVectorSize = index2tick.count

class Drummer {
  let Wx_data: Data
  let Wy_data: Data

  var c = [Float](repeating: 0, count: hiddenSize)
  var h = [Float](repeating: 0, count: hiddenSize + 1)

  init() {
    let Wx_url = Bundle.main.url(forResource: "Wx", withExtension: "bin")!
    Wx_data = try! Data(contentsOf: Wx_url)

    let Wy_url = Bundle.main.url(forResource: "Wy", withExtension: "bin")!
    Wy_data = try! Data(contentsOf: Wy_url)
  }

  /*
    Returns a new list of (note, ticks) pairs.
  */
  func sample(_ n: Int) -> [(Int, Int)] {
    var seedIndexNote = Math.random(noteVectorSize)
    var seedIndexTick = Math.random(tickVectorSize)

    // Start with a random memory.
    Math.uniformRandom(&h, hiddenSize, 0.1)
    Math.uniformRandom(&c, hiddenSize, 0.1)

    // Working space.
    var x = [Float](repeating: 0, count: noteVectorSize + tickVectorSize + hiddenSize + 1)
    var y = [Float](repeating: 0, count: noteVectorSize + tickVectorSize)
    var gates = [Float](repeating: 0, count: hiddenSize*4)
    var tmp = [Float](repeating: 0, count: hiddenSize)

    var sampled: [(Int, Int)] = []

    for _ in 0..<n {
      // One-hot encode the input values for the notes and ticks separately.
      x[seedIndexNote] = 1
      x[seedIndexTick + noteVectorSize] = 1

      // Copy the h vector into x.
      x.withUnsafeMutableBufferPointer { buf in
        let ptr = buf.baseAddress!.advanced(by: noteVectorSize + tickVectorSize)
        memcpy(ptr, &h, hiddenSize * MemoryLayout<Float>.stride)
      }

      // Set the last element to 1 for the bias.
      x[x.count - 1] = 1

      // Multiply x with Wx.
      Wx_data.withUnsafeBytes { Wx in
        Math.matmul(&x, Wx, &gates, 1, Wx_cols, Wx_rows)
      }

      gates.withUnsafeMutableBufferPointer { ptr in
        let gateF = ptr.baseAddress!
        let gateI = gateF.advanced(by: hiddenSize)
        let gateO = gateI.advanced(by: hiddenSize)
        let gateG = gateO.advanced(by: hiddenSize)

        // Compute the activations of the gates.
        Math.sigmoid(gateF, hiddenSize*3)
        Math.tanh(gateG, hiddenSize)

        // c[t] = gateF * sigmoid(c[t-1]) + sigmoid(gateI) * tanh(gateG)
        Math.multiply(&c, gateF, &c, hiddenSize)
        Math.multiply(gateI, gateG, &tmp, hiddenSize)
        Math.add(&tmp, &c, hiddenSize)

        // h[t] = sigmoid(gateO) * tanh(c[t])
        Math.tanh(&c, &tmp, hiddenSize)
        Math.multiply(gateO, &tmp, &h, hiddenSize)
      }

      // Set the last element to 1 for the bias.
      h[h.count - 1] = 1

      // Multiply h with Wy to get y.
      Wy_data.withUnsafeBytes { Wy in
        Math.matmul(&h, Wy, &y, 1, Wy_cols, Wy_rows)
      }

      // Predict the next note and duration.
      Math.softmax(&y, noteVectorSize)
      Math.softmax(&y[noteVectorSize], tickVectorSize)

      // Randomly sample from the output probability distributions.
      let noteIndex = Math.randomlySample(&y, noteVectorSize)
      let tickIndex = Math.randomlySample(&y[noteVectorSize], tickVectorSize)
      sampled.append((index2note[noteIndex], index2tick[tickIndex]))

      // Use the output as the next input.
      x[seedIndexNote] = 0
      x[seedIndexTick + noteVectorSize] = 0
      seedIndexNote = noteIndex
      seedIndexTick = tickIndex
    }

    return sampled
  }
}
