ObjC.import('Foundation');
ObjC.import('PDFKit');
function run(argv) {
  var result=[];
  argv.forEach(function(path) {
    var doc=$.PDFDocument.alloc.initWithURL($.NSURL.fileURLWithPath(path));
    if (!doc) throw new Error('Cannot open PDF: '+path);
    var pages=[];
    for (var i=0;i<doc.pageCount;i++) pages.push({page:i+1,text:ObjC.unwrap(doc.pageAtIndex(i).string)||''});
    result.push({path:path,page_count:Number(doc.pageCount),pages:pages});
  });
  return JSON.stringify(result);
}
