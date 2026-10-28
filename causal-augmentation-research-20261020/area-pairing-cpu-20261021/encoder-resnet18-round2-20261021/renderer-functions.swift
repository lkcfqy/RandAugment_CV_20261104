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

