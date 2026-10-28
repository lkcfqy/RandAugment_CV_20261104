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
