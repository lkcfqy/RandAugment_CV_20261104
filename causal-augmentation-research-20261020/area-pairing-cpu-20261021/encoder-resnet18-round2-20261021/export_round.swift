import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import CryptoKit
import Darwin

struct Raster { var w:Int; var h:Int; var bytes:[UInt8] }
let pixelFlags = CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
func image(_ r:Raster)->CGImage {
    let provider=CGDataProvider(data:Data(r.bytes) as CFData)!
    return CGImage(width:r.w,height:r.h,bitsPerComponent:8,bitsPerPixel:32,bytesPerRow:r.w*4,
                   space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:CGBitmapInfo(rawValue:pixelFlags),
                   provider:provider,decode:nil,shouldInterpolate:true,intent:.defaultIntent)!
}
func raster(_ im:CGImage,_ w:Int,_ h:Int)->Raster {
    precondition(w>0 && h>0 && w*h<=16000000)
    var r=Raster(w:w,h:h,bytes:[UInt8](repeating:0,count:w*h*4))
    r.bytes.withUnsafeMutableBytes { b in
        let c=CGContext(data:b.baseAddress,width:w,height:h,bitsPerComponent:8,bytesPerRow:w*4,
                        space:CGColorSpaceCreateDeviceRGB(),bitmapInfo:pixelFlags)!
        c.interpolationQuality = .high
        c.draw(im,in:CGRect(x:0,y:0,width:w,height:h))
    }
    return r
}
func load(_ url:URL)->Raster {
    let source=CGImageSourceCreateWithURL(url as CFURL,nil)!
    let im=CGImageSourceCreateImageAtIndex(source,0,nil)!
    return raster(im,im.width,im.height)
}
func crop(_ r:Raster,_ x:Int,_ y:Int,_ w:Int,_ h:Int)->Raster {
    precondition(x>=0 && y>=0 && w>0 && h>0 && x+w<=r.w && y+h<=r.h)
    var v=Raster(w:w,h:h,bytes:[UInt8](repeating:0,count:w*h*4))
    for row in 0..<h {
        let start=((y+row)*r.w+x)*4
        v.bytes.replaceSubrange(row*w*4..<(row+1)*w*4,with:r.bytes[start..<start+w*4])
    }
    return v
}
func png(_ r:Raster,_ url:URL) {
    precondition(!FileManager.default.fileExists(atPath:url.path))
    let dest=CGImageDestinationCreateWithURL(url as CFURL,UTType.png.identifier as CFString,1,nil)!
    CGImageDestinationAddImage(dest,image(r),nil); precondition(CGImageDestinationFinalize(dest))
}
func json(_ value:Any,_ url:URL)throws {
    precondition(!FileManager.default.fileExists(atPath:url.path))
    try JSONSerialization.data(withJSONObject:value,options:[.prettyPrinted,.sortedKeys]).write(to:url)
}
func hash(_ bytes:[UInt8])->String { SHA256.hash(data:Data(bytes)).map{String(format:"%02x",$0)}.joined() }

func largestBackground(_ m:Raster)->[Int] {
    var heights=[Int](repeating:0,count:m.w), best=[0,0,0,0], area=0
    for y in 0..<m.h {
        for x in 0..<m.w { heights[x] = m.bytes[(y*m.w+x)*4]==2 ? heights[x]+1 : 0 }
        var stack=[Int]()
        for x in 0...m.w {
            let current = x==m.w ? 0 : heights[x]
            while let last=stack.last, heights[last]>current {
                let h=heights[stack.removeLast()]
                let start=(stack.last ?? -1)+1, w=x-start
                if w*h>area {area=w*h;best=[start,y-h+1,w,h]}
            }
            if x<m.w { stack.append(x) }
        }
    }
    return best
}

struct RandomStream {
    var state:UInt64
    init(_ key:String) {
        let digest=Array(SHA256.hash(data:Data(key.utf8)))
        state=digest.prefix(8).enumerated().reduce(UInt64(0)){$0 | UInt64($1.element)<<($1.offset*8)}
    }
    mutating func unit()->Double {
        state &+= 0x9e3779b97f4a7c15
        var z=state;z=(z^(z>>30)) &* 0xbf58476d1ce4e5b9;z=(z^(z>>27)) &* 0x94d049bb133111eb
        z=z^(z>>31);return Double(z>>11)/9007199254740992.0
    }
}
struct PhotoSource {
    var id:String;var species:Int;var split:String;var role:String;var block:Int;var ratio:Double
    var pixels:Raster
}
struct PreparedForeground {
    var pixels:Raster;var guardMask:[UInt8];var t:Double;var cropped:Bool;var area:Double
    var retained:Double;var sigma:Double
}
func blurAlpha(_ pixels:Raster,_ sigma:Double)->[Double] {
    let w=pixels.w,h=pixels.h,radius=Int(ceil(3*sigma))
    var kernel=(-radius...radius).map{exp(-Double($0*$0)/(2*sigma*sigma))}
    let total=kernel.reduce(0,+);kernel=kernel.map{$0/total}
    var temp=[Double](repeating:0,count:w*h),result=temp
    for y in 0..<h {for x in 0..<w {
        var v=0.0
        for k in -radius...radius {v += kernel[k+radius]*Double(pixels.bytes[(y*w+max(0,min(w-1,x+k)))*4+3])}
        temp[y*w+x]=v
    }}
    for y in 0..<h {for x in 0..<w {
        var v=0.0
        for k in -radius...radius {v += kernel[k+radius]*temp[max(0,min(h-1,y+k))*w+x]}
        result[y*w+x]=v
    }}
    return result
}
func prepare(_ foreground:PhotoSource,_ base:PhotoSource)->PreparedForeground {
    var random=RandomStream("pets-scale-transfer-v1:"+foreground.id+":"+base.id)
    let lo=0.7*min(foreground.ratio,base.ratio), hi=1.3*max(foreground.ratio,base.ratio)
    let t=lo+random.unit()*(hi-lo)
    let f=foreground.pixels
    let fill=Double(stride(from:3,to:f.bytes.count,by:4).reduce(0){$0+Int(f.bytes[$1])})/Double(f.w*f.h*255)
    let scale=t/fill
    let gw=max(1,Int(sqrt(16384*scale*Double(f.w)/Double(f.h)).rounded(.toNearestOrEven)))
    let gh=max(1,Int(sqrt(16384*scale*Double(f.h)/Double(f.w)).rounded(.toNearestOrEven)))
    let scaled=raster(image(f),gw,gh),cw=min(128,gw),ch=min(128,gh)
    let cx=gw>cw ? min(gw-cw,Int(random.unit()*Double(gw-cw+1))) : 0
    let cy=gh>ch ? min(gh-ch,Int(random.unit()*Double(gh-ch+1))) : 0
    let small=crop(scaled,cx,cy,cw,ch)
    let ox=Int((random.unit()*Double(128-cw)).rounded(.toNearestOrEven))
    let oy=Int((random.unit()*Double(128-ch)).rounded(.toNearestOrEven))
    let sigma=0.4+3.6*random.unit(), blurred=blurAlpha(small,sigma)
    var out=Raster(w:128,h:128,bytes:[UInt8](repeating:0,count:128*128*4)), area=0.0
    for y in 0..<ch {for x in 0..<cw {
        let si=(y*cw+x)*4,di=((y+oy)*128+x+ox)*4
        let p=max(0,min(255,Int(blurred[y*cw+x].rounded())))
        let mask=p>128 ? 2*p-255 : 0
        area += Double(mask)/255
        out.bytes[di+3]=UInt8(mask)
        let a=Double(small.bytes[si+3])
        for c in 0..<3 {
            let rgb=a>0 ? min(255,Double(small.bytes[si+c])*255/a) : 0
            out.bytes[di+c]=UInt8(max(0,min(mask,Int((rgb*Double(mask)/255).rounded()))))
        }
    }}
    var horizontal=[UInt8](repeating:0,count:16384),guardMask=horizontal
    for y in 0..<128 {for x in 0..<128 {
        for dx in -3...3 where x+dx>=0 && x+dx<128 {
            if out.bytes[(y*128+x+dx)*4+3]>0 {horizontal[y*128+x]=1;break}
        }
    }}
    for y in 0..<128 {for x in 0..<128 {
        for dy in -3...3 where y+dy>=0 && y+dy<128 {
            if horizontal[(y+dy)*128+x]>0 {guardMask[y*128+x]=1;break}
        }
    }}
    return PreparedForeground(pixels:out,guardMask:guardMask,t:t,cropped:gw>128 || gh>128,
                              area:area/16384,retained:area/(16384*t),sigma:sigma)
}
func combine(_ f:PreparedForeground,_ b:Raster,_ protected:Bool)->Raster {
    var out=b
    for i in 0..<16384 {
        let p=i*4,a=Double(f.pixels.bytes[p+3])/255
        for c in 0..<3 {
            let background=protected && f.guardMask[i]>0 ? 128 : Int(b.bytes[p+c])
            out.bytes[p+c]=UInt8(max(0,min(255,Int((Double(f.pixels.bytes[p+c])+Double(background)*(1-a)).rounded()))))
        }
        out.bytes[p+3]=255
    }
    return out
}
func features(_ r:Raster)->[Double] {
    precondition(r.w==128 && r.h==128)
    var luminance=[Double](repeating:0,count:16384),v=[Double](repeating:0,count:64)
    for i in 0..<16384 {
        let p=i*4
        luminance[i]=(0.2126*Double(r.bytes[p])+0.7152*Double(r.bytes[p+1])+0.0722*Double(r.bytes[p+2]))/255
    }
    for y in 0..<128 {for x in 0..<128 {
        let i=y*128+x,p=i*4,cell=((y/32)*4+x/32)*4
        for c in 0..<3 {v[cell+c] += Double(r.bytes[p+c])/(255*1024)}
        let dx=x<127 ? abs(luminance[i]-luminance[i+1]) : 0
        let dy=y<127 ? abs(luminance[i]-luminance[i+128]) : 0
        v[cell+3] += (dx+dy)/(2*1024)
    }}
    return v
}
func guardHash(_ r:Raster,_ mask:[UInt8])->String {
    var values=[UInt8]();values.reserveCapacity(mask.reduce(0){$0+Int($1)}*3)
    for i in 0..<16384 where mask[i]>0 {values.append(contentsOf:r.bytes[i*4..<i*4+3])}
    return hash(values)
}
func maskRaster(_ values:[UInt8])->Raster {
    var r=Raster(w:128,h:128,bytes:[UInt8](repeating:255,count:65536))
    for i in 0..<16384 {let v=values[i]>0 ? UInt8(255) : UInt8(0);for c in 0..<3 {r.bytes[i*4+c]=v}}
    return r
}
func appendJSON(_ row:[String:Any],_ file:FileHandle)throws {
    var data=try JSONSerialization.data(withJSONObject:row,options:[.sortedKeys]);data.append(10);try file.write(contentsOf:data)
}

// Concatenated after the unchanged original renderer functions.
let args=CommandLine.arguments
setbuf(stdout,nil)
precondition(args.count==4,"export-round calibration|main DATA_ROOT ROUND_ROOT")
let mode=args[1], dataRoot=URL(fileURLWithPath:args[2],isDirectory:true)
let roundRoot=URL(fileURLWithPath:args[3],isDirectory:true)
precondition(mode=="calibration" || mode=="main")
if mode=="main" {precondition(FileManager.default.fileExists(atPath:roundRoot.appendingPathComponent("measurement-freeze.json").path))}
_ = nice(10);alarm(120)
let started=Date()
let plan=try JSONSerialization.jsonObject(with:Data(contentsOf:dataRoot.appendingPathComponent("pixel-photo-plan.json"))) as! [String:Any]
let allRows=plan["rows"] as! [[String:Any]]
let rows=allRows.filter{mode=="main" || ($0["split"] as! String)=="fit"}
let oldCases=try JSONSerialization.jsonObject(with:Data(contentsOf:dataRoot.appendingPathComponent("render-cases.json"))) as! [String:Any]
let reference=Dictionary(uniqueKeysWithValues:(oldCases["rows"] as! [[String:Any]]).map{($0["case"] as! Int,$0)})
let calibrationPlan=try JSONSerialization.jsonObject(with:Data(contentsOf:roundRoot.appendingPathComponent("source-train-calibration-plan.json"))) as! [String:Any]
let calRows=calibrationPlan["calibration_sources"] as! [[String:Any]]
let partitions=Dictionary(uniqueKeysWithValues:calRows.map{($0["source_id"] as! String,$0["partition"] as! String)})
var sources=[PhotoSource](),sourceChecks=[[String:Any]]()
for row in rows {
    let id=row["source_id"] as! String,role=row["role"] as! String
    let photo=load(dataRoot.appendingPathComponent("photos/"+id+".jpg"))
    let mask=load(dataRoot.appendingPathComponent("masks/"+id+".png"))
    precondition(photo.w==mask.w && photo.h==mask.h)
    var pixels:Raster
    if role=="foreground" {
        var masked=photo
        for i in 0..<photo.w*photo.h where mask.bytes[i*4] != 1 {for c in 0..<4 {masked.bytes[i*4+c]=0}}
        let bb=row["foreground_bbox"] as! [Int]
        pixels=crop(masked,bb[0],bb[1],bb[2],bb[3])
    } else {
        let bb=row["largest_background_rectangle"] as! [Int]
        for y in bb[1]..<bb[1]+bb[3] {for x in bb[0]..<bb[0]+bb[2] {precondition(mask.bytes[(y*mask.w+x)*4]==2)}}
        pixels=raster(image(crop(photo,bb[0],bb[1],bb[2],bb[3])),128,128)
    }
    sources.append(PhotoSource(id:id,species:row["species"] as! Int,split:row["split"] as! String,
        role:role,block:row["macroblock"] as! Int,ratio:row["ratio_input"] as! Double,pixels:pixels))
    sourceChecks.append(["id":id,"role":role,"split":row["split"]!,"decoded_RGBA_sha256":hash(photo.bytes)])
}
var gray=Raster(w:128,h:128,bytes:[UInt8](repeating:128,count:65536))
for p in stride(from:3,to:gray.bytes.count,by:4){gray.bytes[p]=255}
let rawURL=roundRoot.appendingPathComponent(mode+"-frames.rgba")
precondition(!FileManager.default.fileExists(atPath:rawURL.path))
precondition(FileManager.default.createFile(atPath:rawURL.path,contents:nil))
let stream=try FileHandle(forWritingTo:rawURL)
var records=[[String:Any]](),index=0,caseID=0,geometryChecks=0,guardChecks=0
var maxFeatureReadbackError=0.0,maxTReadbackError=0.0
func emit(_ pixels:Raster,_ metadata:[String:Any])throws {
    precondition(pixels.w==128 && pixels.h==128)
    try stream.write(contentsOf:Data(pixels.bytes))
    var row=metadata;row["index"]=index;row["RGBA_sha256"]=hash(pixels.bytes)
    records.append(row);index+=1
}
for split in (mode=="calibration" ? ["fit"] : ["fit","heldout"]) {
    for block in (split=="fit" ? [-1] : [0,1,2,3]) {
        let fg=sources.filter{$0.split==split && $0.role=="foreground" && $0.block==block}
        let bg=sources.filter{$0.split==split && $0.role=="background" && $0.block==block}
        precondition(fg.count==(split=="fit" ? 16:4) && bg.count==fg.count)
        for f in fg {for base in bg {
            let prepared=prepare(f,base),neutral=combine(prepared,gray,false)
            let old=reference[caseID]!
            precondition((old["foreground_id"] as! String)==f.id && (old["base_background_id"] as! String)==base.id)
            let tError=abs((old["T"] as! Double)-prepared.t)
            let featureError=zip(old["neutral_foreground_features"] as! [Double],features(neutral)).map{abs($0-$1)}.max()!
            if caseID==0 {print("JSON readback check firstcase T_abs_error=\(tError) feature_max_abs_error=\(featureError)")}
            maxFeatureReadbackError=max(maxFeatureReadbackError,featureError);maxTReadbackError=max(maxTReadbackError,tError)
            precondition(tError<=1e-14 && (old["cropped"] as! Bool)==prepared.cropped,"T/crop mismatch case \(caseID) error \(tError)")
            precondition((old["mask_sha256"] as! String)==hash(stride(from:3,to:prepared.pixels.bytes.count,by:4).map{prepared.pixels.bytes[$0]}))
            precondition(featureError<=1e-14,"feature mismatch case \(caseID) error \(featureError)")
            geometryChecks+=1
            var meta:[String:Any]=["case":caseID,"split":split,"block":block,"foreground_id":f.id,
                "base_background_id":base.id,"Y":f.species,"T":prepared.t,"cropped":prepared.cropped,
                "visible_alpha_fraction":prepared.area,"retained_fraction_of_T":prepared.retained]
            if mode=="calibration" {
                meta["kind"]="scaled_neutral";meta["partition"]=partitions[f.id]!
                try emit(neutral,meta)
            } else {
                var expectedGuard:String?=nil
                for actual in bg {for protected in [false,true] {
                    let pixels=combine(prepared,actual.pixels,protected)
                    var row=meta;row["kind"]="composite";row["layout"]=protected ? "protected":"ordinary"
                    row["actual_background_id"]=actual.id;row["C"]=actual.species;row["coupled"]=actual.id==base.id
                    if protected {
                        let digest=guardHash(pixels,prepared.guardMask)
                        if let expected=expectedGuard {precondition(digest==expected)} else {expectedGuard=digest}
                        guardChecks+=1
                    }
                    try emit(pixels,row)
                }}
            }
            caseID+=1
        }}
        print("export mode=\(mode) split=\(split) block=\(block) frames=\(index)")
    }
}
if mode=="calibration" {
    for f in sources where f.role=="foreground" && partitions[f.id]=="calibration" {
        let scale=96.0/Double(max(f.pixels.w,f.pixels.h))
        let w=max(1,Int((Double(f.pixels.w)*scale).rounded(.toNearestOrEven)))
        let h=max(1,Int((Double(f.pixels.h)*scale).rounded(.toNearestOrEven)))
        let small=raster(image(f.pixels),w,h),ox=(128-w)/2,oy=(128-h)/2
        var pixels=gray
        for y in 0..<h {for x in 0..<w {
            let si=(y*w+x)*4,di=((oy+y)*128+ox+x)*4,a=Double(small.bytes[si+3])/255
            for c in 0..<3 {pixels.bytes[di+c]=UInt8(max(0,min(255,Int((Double(small.bytes[si+c])+128*(1-a)).rounded()))))}
        }}
        try emit(pixels,["kind":"intact_neutral","partition":"calibration","foreground_id":f.id,"Y":f.species,"split":"fit"])
        png(pixels,roundRoot.appendingPathComponent("intact-"+f.id+".png"))
    }
}
try stream.close()
precondition(index==(mode=="calibration" ? 260:8704))
try json(["mode":mode,"records":records,"frames":index,"RGBA_frame_bytes":65536,"width":128,"height":128,
    "source_checks":sourceChecks,"original_case_mask_hash_exact_matches":geometryChecks,
    "original_neutral_feature_max_abs_JSON_readback_error":maxFeatureReadbackError,
    "original_T_max_abs_JSON_readback_error":maxTReadbackError,"JSON_numeric_tolerance":1e-14,
    "protected_guard_hash_checks":guardChecks,"CPU_only":true,"GPU":false,
    "wall_seconds":Date().timeIntervalSince(started)],roundRoot.appendingPathComponent(mode+"-frames.json"))
print("completed mode=\(mode) frames=\(index) seconds=\(Date().timeIntervalSince(started))")
