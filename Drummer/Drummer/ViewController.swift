import UIKit
import AVFoundation

class ViewController: UIViewController {
  @IBOutlet weak var button: UIButton!

  var midiPlayer: AVMIDIPlayer?
  let drummer = Drummer()
  let soundFont = "1115-Standard Rock Set"

  override func viewDidLoad() {
    super.viewDidLoad()

    do {
		try AVAudioSession.sharedInstance().setCategory(AVAudioSession.Category.playback)
    } catch {
      print("Error: could not set audio session")
    }
  }

  @IBAction func playDrums(_ sender: Any) {
    if midiPlayer == nil {
      button.setTitle("Stop this madness", for: .normal)
      hitMeWithYourRhythmStick()
    } else {
      button.setTitle("Play another one!", for: .normal)
      midiPlayer?.stop()
      midiPlayer = nil
    }
  }

  func hitMeWithYourRhythmStick() {
    let drums = drummer.sample(1000)

    if let sequence = createMusicSequence(drums),
      let data = dataFromMusicSequence(sequence) {

      let midiData = data.takeUnretainedValue() as Data
      midiPlayer = createMIDIPlayer(midiData: midiData)
      data.release()

      midiPlayer?.play(nil)
    }
  }

  func createMusicSequence(_ drums: [(Int, Int)]) -> MusicSequence? {
    var musicSequence: MusicSequence?
    var status = NewMusicSequence(&musicSequence)
    guard status == OSStatus(noErr) else {
      print("Error: could not create MusicSequence \(status)")
      return nil
    }

    var track: MusicTrack?
    status = MusicSequenceNewTrack(musicSequence!, &track)
    guard status == OSStatus(noErr) else {
      print("Error: could not create MusicTrack \(status)")
      return nil
    }

    var totalTicks = 0
    for (note, ticks) in drums {
      var message = MIDINoteMessage(channel: 10, note: UInt8(note),
                                    velocity: 64, releaseVelocity: 0,
                                    duration: 0.5)

      totalTicks += ticks
      let beat = MusicTimeStamp(totalTicks) / 480

      status = MusicTrackNewMIDINoteEvent(track!, beat, &message)
      if status != OSStatus(noErr) {
        print("Error: could not create MIDINoteEvent \(status)")
      }
    }

    //CAShow(UnsafeMutablePointer<MusicSequence>(musicSequence!))

    return musicSequence!
  }

  func dataFromMusicSequence(_ sequence: MusicSequence) -> Unmanaged<CFData>? {
    var data: Unmanaged<CFData>?
    let status = MusicSequenceFileCreateData(sequence, .midiType, .eraseFile, 480, &data)
    guard status == OSStatus(noErr) else {
      print("Error: could not create data from MusicSequence \(status)")
      return nil
    }
    return data
  }

  func createMIDIPlayer(midiData: Data) -> AVMIDIPlayer? {
    guard let url = Bundle.main.url(forResource: soundFont, withExtension: "sf2") else {
      print("Error: could not load \(soundFont)")
      return nil
    }

    do {
      let midiPlayer = try AVMIDIPlayer(data: midiData, soundBankURL: url)
      midiPlayer.prepareToPlay()
      return midiPlayer
    } catch {
      print("Error: could not create AVMIDIPlayer \(error)")
      return nil
    }
  }
}
