import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import Darwin

struct Raster {
    var w: Int
    var h: Int
    var bytes: [UInt8]
}
let flags = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
func cgImage(_ r: Raster) -> CGImage {
    let data = Data(r.bytes) as CFData
    let provider = CGDataProvider(data: data)!
    return CGImage(width:r.w, height:r.h, bitsPerComponent:8, bitsPerPixel:32,
                   bytesPerRow:r.w*4, space:CGColorSpaceCreateDeviceRGB(),
                   bitmapInfo:CGBitmapInfo(rawValue:flags), provider:provider,
                   decode:nil, shouldInterpolate:true, intent:.defaultIntent)!
}
func raster(_ im: CGImage, _ w: Int, _ h: Int) -> Raster {
    var bytes = [UInt8](repeating:0, count:w*h*4)
    bytes.withUnsafeMutableBytes { buf in
        let ctx = CGContext(data:buf.baseAddress, width:w, height:h, bitsPerComponent:8,
                            bytesPerRow:w*4, space:CGColorSpaceCreateDeviceRGB(), bitmapInfo:flags)!
        ctx.interpolationQuality = .high
        ctx.draw(im, in:CGRect(x:0,y:0,width:w,height:h))
    }
    return Raster(w:w,h:h,bytes:bytes)
}
func writePNG(_ r: Raster, _ url: URL) {
    precondition(!FileManager.default.fileExists(atPath:url.path), "Preserving existing image")
    let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)!
    CGImageDestinationAddImage(dest,cgImage(r),nil)
    precondition(CGImageDestinationFinalize(dest))
}
func crop(_ r: Raster, _ x: Int, _ y: Int, _ w: Int, _ h: Int) -> Raster {
    precondition(x>=0 && y>=0 && x+w<=r.w && y+h<=r.h)
    var out=Raster(w:w,h:h,bytes:[UInt8](repeating:0,count:w*h*4))
    for row in 0..<h {
        let start=((row+y)*r.w+x)*4
        out.bytes.replaceSubrange(row*w*4..<(row+1)*w*4,with:r.bytes[start..<start+w*4])
    }
    return out
}
func inside(_ x: Double, _ y: Double, _ polygon: [[Double]]) -> Bool {
    var result=false
    var j=polygon.count-1
    for i in 0..<polygon.count {
        let a=polygon[i],b=polygon[j]
        if ((a[1]>y) != (b[1]>y)) && x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0] { result.toggle() }
        j=i
    }
    return result
}
func darkFraction(_ r: Raster) -> Double {
    var count=0
    for p in stride(from:0,to:r.bytes.count,by:4) {
        let luma=0.2126*Double(r.bytes[p])+0.7152*Double(r.bytes[p+1])+0.0722*Double(r.bytes[p+2])
        if luma<128 { count += 1 }
    }
    return Double(count)/Double(r.w*r.h)
}
func alphaArea(_ r: Raster) -> Double {
    var value=0.0
    for p in stride(from:3,to:r.bytes.count,by:4) { value += Double(r.bytes[p])/255.0 }
    return value
}
func opaque(_ w: Int, _ h: Int) -> Raster {
    return Raster(w:w,h:h,bytes:[UInt8](repeating:255,count:w*h*4))
}
func paste(_ fg: Raster, _ bg: Raster, _ x: Int, _ y: Int) -> Raster {
    var out=bg
    for row in 0..<fg.h {
        for col in 0..<fg.w {
            let si=(row*fg.w+col)*4, di=((row+y)*bg.w+col+x)*4
            let a=Double(fg.bytes[si+3])/255.0
            for c in 0..<3 {
                out.bytes[di+c]=UInt8(max(0,min(255,Int((Double(fg.bytes[si+c])+Double(bg.bytes[di+c])*(1-a)).rounded()))))
            }
            out.bytes[di+3]=255
        }
    }
    return out
}
func jsonWrite(_ value: Any, _ url: URL) throws {
    precondition(!FileManager.default.fileExists(atPath:url.path), "Preserving existing JSON")
    try JSONSerialization.data(withJSONObject:value,options:[.prettyPrinted,.sortedKeys]).write(to:url)
}
let args=CommandLine.arguments
precondition(args.count==3,"usage: pixel-check photo.jpg output-dir")
let sourceURL=URL(fileURLWithPath:args[1])
let out=URL(fileURLWithPath:args[2],isDirectory:true)
let imageDir=out.appendingPathComponent("images",isDirectory:true)
precondition(!FileManager.default.fileExists(atPath:imageDir.path),"Preserving existing output")
try FileManager.default.createDirectory(at:imageDir,withIntermediateDirectories:false)
_ = nice(10)
let start=Date()
let source=CGImageSourceCreateWithURL(sourceURL as CFURL,nil)!
let im=CGImageSourceCreateImageAtIndex(source,0,nil)!
let original=raster(im,im.width,im.height)
precondition(original.w==960 && original.h==587)
let polygon:[[Double]]=[
 [51,343],[65,307],[97,278],[144,248],[196,239],[310,232],[412,239],[515,251],
 [566,243],[594,224],[607,184],[611,127],[633,139],[651,164],[711,164],[740,138],
 [765,122],[768,165],[758,217],[758,258],[743,289],[771,315],[812,327],[856,313],
 [884,308],[907,316],[920,337],[918,358],[897,370],[832,371],[756,367],[635,369],
 [527,367],[433,363],[355,360],[293,359],[248,361],[206,373],[179,384],[151,390],
 [142,406],[157,435],[181,458],[199,475],[207,493],[200,510],[184,513],[161,503],
 [132,481],[106,462],[82,437],[65,410],[55,380]
]
var masked=original
for y in 0..<original.h {
    for x in 0..<original.w {
        if !inside(Double(x)+0.5,Double(y)+0.5,polygon) {
            let p=(y*original.w+x)*4
            for c in 0..<4 { masked.bytes[p+c]=0 }
        }
    }
}
let foreground=crop(masked,45,115,885,410)
writePNG(foreground,imageDir.appendingPathComponent("foreground-manual-outline.png"))
writePNG(original,imageDir.appendingPathComponent("source-roundtrip.png"))
let alphaFactor=alphaArea(foreground)/Double(foreground.w*foreground.h)
precondition(alphaFactor>0.2 && alphaFactor<0.95)
let bg0=raster(cgImage(crop(original,0,0,480,100)),256,256)
let bg1=raster(cgImage(crop(original,400,400,560,187)),256,256)
writePNG(bg0,imageDir.appendingPathComponent("background-0.png"))
writePNG(bg1,imageDir.appendingPathComponent("background-1.png"))
let trainT:[Double]=[0.03,0.07,0.11,0.15]
let testT:[Double]=[0.05,0.09,0.13,0.17]
let probeT:[Double]=[0.22,0.35,0.5,0.7,0.9,1.2]
let levels=(trainT+testT+probeT).sorted()
var records:[[String:Any]]=[]
var previews:[Raster]=[]
for (bgi,bg) in [bg0,bg1].enumerated() {
    for t in levels {
        let scale=t/alphaFactor
        let w=max(1,Int((sqrt(65536.0*scale*Double(foreground.w)/Double(foreground.h))).rounded(.toNearestOrEven)))
        let h=max(1,Int((sqrt(65536.0*scale*Double(foreground.h)/Double(foreground.w))).rounded(.toNearestOrEven)))
        let scaled=raster(cgImage(foreground),w,h)
        let cropped=crop(scaled,max(0,(w-min(w,256))/2),max(0,(h-min(h,256))/2),min(w,256),min(h,256))
        let composite=paste(cropped,bg,(256-cropped.w)/2,(256-cropped.h)/2)
        let file=String(format:"bg%d-T%.2f.png",bgi,t)
        writePNG(composite,imageDir.appendingPathComponent(file))
        let role=(bgi==0 && trainT.contains(t)) ? "fit" : ((bgi==1 && testT.contains(t)) ? "heldout-T-and-bg" : "probe-or-unused")
        records.append(["background":bgi,"target_T":t,"goal_width":w,"goal_height":h,
                        "was_cropped":w>256 || h>256,"visible_alpha_fraction":alphaArea(cropped)/65536.0,
                        "dark_fraction_image_only":darkFraction(composite),"role":role,"file":"images/"+file])
        if [0.05,0.17,0.5,1.2].contains(t) { previews.append(composite) }
    }
}
var montage=opaque(1024,512)
for (i,img) in previews.enumerated() { montage=paste(img,montage,(i%4)*256,(i/4)*256) }
writePNG(montage,out.appendingPathComponent("pixel-carrier-contact-sheet.png"))
try jsonWrite(["manual_polygon_source_coordinates":polygon,"foreground_crop":[45,115,885,410],
               "background_crops":[[0,0,480,100],[400,400,560,187]],"alpha_factor":alphaFactor,
               "coordinate_note":"Coordinates selected from viewed existing photograph; inspect generated cutout before accepting measurements."],out.appendingPathComponent("mask-and-crops.json"))
let result:[String:Any]=[
 "status":"rendered_pending_visual_and_numeric_checks","records":records,
 "photo_dimensions":[original.w,original.h],"manual_alpha_factor":alphaFactor,
 "render_backend":"macOS CoreGraphics/ImageIO software bitmap; high interpolation",
 "pixel_exact_ForAug_reproduction":false,"crop_mode":"center crop (eval branch), centered placement",
 "photographs":1,"background_crops":2,"composites":records.count,
 "training_or_inference_of_vision_models":false,"ARIS":false,"GPU_API_calls":false,
 "new_software_installed":false,"new_photos_downloaded":false,
 "wall_seconds":Date().timeIntervalSince(start),
 "limits":"One reused real cat photograph with rough manual outline and same-photo background crops. Not actual ForNet masks, independent classes, semantic-background labels, unseen objects or visual-shortcut evidence."
]
try jsonWrite(result,out.appendingPathComponent("pixel-raw-results.json"))
print(String(data:try JSONSerialization.data(withJSONObject:["composites":records.count,"alpha_factor":alphaFactor,"wall_seconds":Date().timeIntervalSince(start)]),encoding:.utf8)!)
