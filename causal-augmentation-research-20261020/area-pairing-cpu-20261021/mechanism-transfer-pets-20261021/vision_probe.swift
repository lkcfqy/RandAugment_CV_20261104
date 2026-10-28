import Foundation
import Vision
import CoreML
import CoreGraphics
import ImageIO
import Darwin

alarm(20); _ = nice(10)
let request=VNGenerateImageFeaturePrintRequest()
request.revision=VNGenerateImageFeaturePrintRequestRevision2
let devices=try request.supportedComputeStageDevices
var assignments=[String:String]()
for (stage,allowed) in devices {
    let cpus=allowed.filter{if case .cpu = $0{return true};return false}
    precondition(!cpus.isEmpty,"CPU not available for a compute stage")
    request.setComputeDevice(cpus[0],for:stage)
    guard let selected=request.computeDevice(for:stage) else {preconditionFailure("No assigned device")}
    guard case .cpu = selected else {preconditionFailure("Non-CPU device")}
    assignments[String(describing:stage)]="CPU"
}
precondition(!assignments.isEmpty)
let source=CGImageSourceCreateWithURL(URL(fileURLWithPath:CommandLine.arguments[1]) as CFURL,nil)!
let im=CGImageSourceCreateImageAtIndex(source,0,nil)!
let started=Date()
try VNImageRequestHandler(cgImage:im,options:[:]).perform([request])
let observation=request.results!.first!
let result:[String:Any]=["status":"existing_native_featureprint_CPU_probe_completed",
                        "revision":request.revision,"assigned_compute_devices":assignments,
                        "element_count":observation.elementCount,"element_type":observation.elementType.rawValue,
                        "data_bytes":observation.data.count,"wall_seconds":Date().timeIntervalSince(started),
                        "new_dependencies_installed":false,"model_download_requested":false,
                        "GPU_or_ANE_device_assigned":false]
let b=try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
print(String(data:b,encoding:.utf8)!)
