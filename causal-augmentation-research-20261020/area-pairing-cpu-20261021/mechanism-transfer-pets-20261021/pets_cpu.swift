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

let args=CommandLine.arguments
precondition(args.count==3,"pets-cpu masks|render ROOT")
let mode=args[1], root=URL(fileURLWithPath:args[2],isDirectory:true)
_ = nice(10); alarm(300)
let started=Date()
if mode=="masks" {
    let plan=try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("mask-pilot-plan.json"))) as! [String:Any]
    let rows=plan["rows"] as! [[String:Any]]
    var result=[[String:Any]]()
    for row in rows {
        let id=row["source_id"] as! String
        precondition(!id.contains("/"))
        let path=root.appendingPathComponent("masks/"+id+".png")
        if !FileManager.default.fileExists(atPath:path.path) {
            var out=row;out["status"]="missing_download";result.append(out);continue
        }
        let m=load(path)
        var histogram=[Int](repeating:0,count:256), gray=true
        var minx=m.w,miny=m.h,maxx = -1,maxy = -1
        for y in 0..<m.h {
            for x in 0..<m.w {
                let i=(y*m.w+x)*4, v=Int(m.bytes[i]);histogram[v]+=1
                if m.bytes[i] != m.bytes[i+1] || m.bytes[i] != m.bytes[i+2] { gray=false }
                if v==1 { minx=min(minx,x);miny=min(miny,y);maxx=max(maxx,x);maxy=max(maxy,y) }
            }
        }
        let values=(0..<256).filter{histogram[$0]>0}
        let valid=gray && values.allSatisfy{[1,2,3].contains($0)} && histogram[1]>0 && histogram[2]>0
        var out=row
        out["width"]=m.w;out["height"]=m.h;out["mask_values"]=values
        out["mask_label_counts"]=["foreground":histogram[1],"background":histogram[2],"unknown":histogram[3]]
        out["decoded_mask_sha256"]=hash(m.bytes)
        out["status"]=valid ? "mask_valid" : "unexpected_mask_encoding"
        if valid {
            let bb=[minx,miny,maxx-minx+1,maxy-miny+1]
            let ratio=Double(bb[2]*bb[3])/Double(m.w*m.h), empty=largestBackground(m)
            out["foreground_bbox"]=bb;out["ratio_input"]=ratio
            out["largest_background_rectangle"]=empty
            out["foreground_eligible"]=histogram[1]>=400 && bb[2]>=16 && bb[3]>=16
            out["background_eligible"]=ratio<0.8 && empty[2]>=64 && empty[3]>=64 && Double(empty[2]*empty[3])/Double(m.w*m.h)>=0.05
        }
        result.append(out)
    }
    try json(["status":"lossless_masks_scanned","rows":result,"wall_seconds":Date().timeIntervalSince(started),
              "software_bitmap":true,"new_software":false,"GPU":false],root.appendingPathComponent("mask-geometry.json"))
    print("mask_rows=\(result.count) valid=\(result.filter{($0["status"] as? String)=="mask_valid"}.count) seconds=\(Date().timeIntervalSince(started))")
} else if mode=="render" {
    let plan=try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent("pixel-photo-plan.json"))) as! [String:Any]
    let rows=plan["rows"] as! [[String:Any]]
    let qa=root.appendingPathComponent("qa",isDirectory:true)
    precondition(!FileManager.default.fileExists(atPath:qa.path))
    try FileManager.default.createDirectory(at:qa,withIntermediateDirectories:false)
    var sources=[PhotoSource](),sourceChecks=[[String:Any]](),hashes=Set<String>()
    for row in rows {
        let id=row["source_id"] as! String,photo=load(root.appendingPathComponent("photos/"+id+".jpg"))
        let m=load(root.appendingPathComponent("masks/"+id+".png")),role=row["role"] as! String
        precondition(photo.w==m.w && photo.h==m.h)
        let digest=hash(photo.bytes);precondition(hashes.insert(digest).inserted,"Exact duplicate photograph")
        var prepared:Raster
        if role=="foreground" {
            var masked=photo
            for i in 0..<photo.w*photo.h where m.bytes[i*4] != 1 {for c in 0..<4 {masked.bytes[i*4+c]=0}}
            let bb=row["foreground_bbox"] as! [Int]
            prepared=crop(masked,bb[0],bb[1],bb[2],bb[3])
        } else {
            let bb=row["largest_background_rectangle"] as! [Int]
            for y in bb[1]..<bb[1]+bb[3] {for x in bb[0]..<bb[0]+bb[2] {precondition(m.bytes[(y*m.w+x)*4]==2)}}
            let small=crop(photo,bb[0],bb[1],bb[2],bb[3]);prepared=raster(image(small),128,128)
        }
        sources.append(PhotoSource(id:id,species:row["species"] as! Int,split:row["split"] as! String,
                                   role:role,block:row["macroblock"] as! Int,ratio:row["ratio_input"] as! Double,pixels:prepared))
        sourceChecks.append(["id":id,"role":role,"species":row["species"]!,"split":row["split"]!,
                             "photo_width":photo.w,"photo_height":photo.h,"decoded_RGBA_sha256":digest,
                             "prepared_width":prepared.w,"prepared_height":prepared.h,
                             "background_rectangle_all_label2":role=="background"])
    }
    precondition(sources.count==64 && hashes.count==64)
    var backgroundRows=[[String:Any]]()
    for s in sources where s.role=="background" {
        backgroundRows.append(["id":s.id,"label":s.species,"split":s.split,"block":s.block,"features":features(s.pixels)])
    }
    try json(["rows":backgroundRows],root.appendingPathComponent("background-source-features.json"))
    let featureURL=root.appendingPathComponent("rgb-features.jsonl")
    precondition(!FileManager.default.fileExists(atPath:featureURL.path))
    precondition(FileManager.default.createFile(atPath:featureURL.path,contents:nil))
    let stream=try FileHandle(forWritingTo:featureURL)
    var cases=[[String:Any]](),caseID=0,featureRows=0,uniqueFrames=0,guardChecks=0
    var qaCases=[[String:Any]](),qaTiles=[Int:[Raster]]()
    var gray=Raster(w:128,h:128,bytes:[UInt8](repeating:128,count:65536))
    for p in stride(from:3,to:gray.bytes.count,by:4){gray.bytes[p]=255}
    for split in ["fit","heldout"] {
        let blocks=split=="fit" ? [-1] : [0,1,2,3]
        for block in blocks {
            let fg=sources.filter{$0.split==split && $0.role=="foreground" && $0.block==block}
            let bg=sources.filter{$0.split==split && $0.role=="background" && $0.block==block}
            precondition(fg.count==(split=="fit" ? 16:4) && bg.count==fg.count)
            for f in fg {
                for base in bg {
                    let prepared=prepare(f,base),neutral=combine(prepared,gray,false)
                    let isQA=split=="heldout" && f.id==fg.first(where:{$0.species==f.species})!.id && base.id==bg[0].id
                    let changed=bg.first(where:{$0.species != base.species})!
                    var expectedGuardHash:String?=nil
                    cases.append(["case":caseID,"split":split,"block":block,"foreground_id":f.id,"base_background_id":base.id,
                                  "Y":f.species,"base_C":base.species,"T":prepared.t,"cropped":prepared.cropped,
                                  "visible_alpha_fraction":prepared.area,"retained_fraction_of_T":prepared.retained,
                                  "mask_sha256":hash(stride(from:3,to:prepared.pixels.bytes.count,by:4).map{prepared.pixels.bytes[$0]}),
                                  "guard_visible_fraction":Double(prepared.guardMask.reduce(0){$0+Int($1)})/16384,
                                  "neutral_foreground_features":features(neutral)])
                    if isQA {
                        png(neutral,qa.appendingPathComponent("case\(caseID)-neutral.png"))
                        png(maskRaster(prepared.guardMask),qa.appendingPathComponent("case\(caseID)-guard.png"))
                    }
                    for actual in bg {
                        for protected in [false,true] {
                            let output=combine(prepared,actual.pixels,protected),vector=features(output)
                            let layout=protected ? "protected" : "ordinary"
                            var metadata:[String:Any]=["case":caseID,"split":split,"block":block,"layout":layout,
                                "Y":f.species,"C":actual.species,"foreground_id":f.id,"base_background_id":base.id,
                                "actual_background_id":actual.id,"features":vector,"policy":"independent"]
                            try appendJSON(metadata,stream);featureRows+=1;uniqueFrames+=1
                            if protected {
                                let h=guardHash(output,prepared.guardMask)
                                if let reference=expectedGuardHash {precondition(h==reference)} else {expectedGuardHash=h}
                                guardChecks+=1
                            }
                            if actual.id==base.id {metadata["policy"]="coupled";try appendJSON(metadata,stream);featureRows+=1}
                            if isQA && (actual.id==base.id || actual.id==changed.id) {
                                let variant=actual.id==base.id ? "coupled" : "changed"
                                let filename="case\(caseID)-\(layout)-\(variant).png"
                                png(output,qa.appendingPathComponent(filename))
                                let slot=(protected ? 2:0)+(variant=="changed" ? 1:0)
                                if qaTiles[caseID]==nil {qaTiles[caseID]=[Raster](repeating:gray,count:4)}
                                qaTiles[caseID]![slot]=output
                                qaCases.append(["case":caseID,"layout":layout,"variant":variant,"file":"qa/"+filename])
                            }
                        }
                    }
                    caseID+=1
                }
            }
            print("rendered split=\(split) block=\(block) cases=\(caseID) uniqueRGB=\(uniqueFrames)")
        }
    }
    try stream.close()
    var sheet=Raster(w:512,h:128*qaTiles.count,bytes:[UInt8](repeating:255,count:512*128*qaTiles.count*4))
    for (row,key) in qaTiles.keys.sorted().enumerated() {
        for col in 0..<4 {let tile=qaTiles[key]![col]
            for y in 0..<128 {let si=y*128*4,di=((row*128+y)*512+col*128)*4
                sheet.bytes.replaceSubrange(di..<di+512,with:tile.bytes[si..<si+512])
            }
        }
    }
    png(sheet,root.appendingPathComponent("mechanism-transfer-contact-sheet.png"))
    try json(["rows":cases],root.appendingPathComponent("render-cases.json"))
    try json(["source_checks":sourceChecks,"QA":qaCases,"cases":caseID,"feature_rows":featureRows,
              "unique_RGB_frames":uniqueFrames,"protected_guard_hash_checks":guardChecks,
              "all_protected_foreground_and_local_edge_pixels_identical_across_BG":true,
              "decoded_source_photographs":64,"exact_duplicate_photographs":0,
              "software_bitmap":true,"GPU":false,"new_dependencies":false,
              "wall_seconds":Date().timeIntervalSince(started),
              "fidelity":"same scale formula, alpha correction, crop/placement and mask-smoothing logic; CoreGraphics resampling and separable Gaussian are not Pillow pixel exact; protected-edge compositor is an explicit visibility control"],root.appendingPathComponent("render-receipt.json"))
    print("render completed featureRows=\(featureRows) uniqueFrames=\(uniqueFrames) seconds=\(Date().timeIntervalSince(started))")
} else {
    preconditionFailure("Unknown mode")
}
