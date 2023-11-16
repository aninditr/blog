const t=JSON.parse('{"key":"v-d12a3a8c","path":"/mt/ctnmt/","title":"\u673A\u5668\u7FFB\u8BD1\u4E2D\u7684 BERT \u5E94\u7528","lang":"en-US","frontmatter":{"title":"\u673A\u5668\u7FFB\u8BD1\u4E2D\u7684 BERT \u5E94\u7528","author":"\u738B\u660E\u8F69","date":"2020-12-31T00:00:00.000Z","tag":["BERT","Pre-training","Catastrophic Forgetting"],"category":["MT"],"summary":"\u200B\\t\u9884\u8BAD\u7EC3\u6280\u672F\uFF0C\u6BD4\u5982 BERT\u7B49\uFF0C\u5728\u81EA\u7136\u8BED\u8A00\u5904\u7406\u9886\u57DF\uFF0C\u5C24\u5176\u662F\u81EA\u7136\u8BED\u8A00\u7406\u89E3\u4EFB\u52A1\u53D6\u5F97\u4E86\u5DE8\u5927\u7684\u6210\u529F\u3002\u7136\u800C\u76EE\u524D\u9884\u8BAD\u7EC3\u6280\u672F\u5728\u6587\u672C\u751F\u6210\u9886\u57DF\uFF0C\u6BD4\u5982\u673A\u5668\u7FFB\u8BD1\u9886\u57DF\uFF0C\u80FD\u591F\u53D6\u5F97\u4EC0\u4E48\u6837\u7684\u6548\u679C\uFF0C\u8FD8\u662F\u4E00\u4E2A\u5F00\u653E\u95EE\u9898\u3002CTNMT \u8FD9\u7BC7\u8BBA\u6587\uFF0C\u4ECE\u4E09\u4E2A\u65B9\u9762\u4ECB\u7ECD\u8FD9\u4E2A\u95EE\u9898\uFF1A\\n\\n\u9884\u8BAD\u7EC3\u6280\u672F\uFF0C\u6BD4\u5982 BERT\u6216\u8005 GPT \u5728\u673A\u5668\u7FFB\u8BD1\u4E2D\u7684\u5E94\u7528\u5B58\u5728\u4EC0\u4E48\u6311\u6218\uFF1F\\n\u9488\u5BF9\u8FD9\u4E9B\u8C03\u6574\uFF0C\u9700\u8981\u600E\u4E48\u6700\u5927\u7A0B\u5EA6\u5229\u7528\u9884\u8BAD\u7EC3\u77E5\u8BC6\uFF1F\\n\u9884\u8BAD\u7EC3\u548C\u673A\u5668\u7FFB\u8BD1\u7684\u878D\u5408\u8FD8\u6709\u4EC0\u4E48\u6F5C\u529B\uFF1F\\n\\n","head":[["meta",{"property":"og:url","content":"https://lileicc.github.io/blog/mt/ctnmt/"}],["meta",{"property":"og:site_name","content":"MLNLP Blog"}],["meta",{"property":"og:title","content":"\u673A\u5668\u7FFB\u8BD1\u4E2D\u7684 BERT \u5E94\u7528"}],["meta",{"property":"og:type","content":"article"}],["meta",{"property":"og:updated_time","content":"2022-10-02T00:01:48.000Z"}],["meta",{"property":"og:locale","content":"en-US"}],["meta",{"property":"article:author","content":"\u738B\u660E\u8F69"}],["meta",{"property":"article:tag","content":"BERT"}],["meta",{"property":"article:tag","content":"Pre-training"}],["meta",{"property":"article:tag","content":"Catastrophic Forgetting"}],["meta",{"property":"article:published_time","content":"2020-12-31T00:00:00.000Z"}],["meta",{"property":"article:modified_time","content":"2022-10-02T00:01:48.000Z"}]]},"excerpt":"<p>\u200B\\t\u9884\u8BAD\u7EC3\u6280\u672F\uFF0C\u6BD4\u5982 BERT\u7B49\uFF0C\u5728\u81EA\u7136\u8BED\u8A00\u5904\u7406\u9886\u57DF\uFF0C\u5C24\u5176\u662F\u81EA\u7136\u8BED\u8A00\u7406\u89E3\u4EFB\u52A1\u53D6\u5F97\u4E86\u5DE8\u5927\u7684\u6210\u529F\u3002\u7136\u800C\u76EE\u524D\u9884\u8BAD\u7EC3\u6280\u672F\u5728\u6587\u672C\u751F\u6210\u9886\u57DF\uFF0C\u6BD4\u5982\u673A\u5668\u7FFB\u8BD1\u9886\u57DF\uFF0C\u80FD\u591F\u53D6\u5F97\u4EC0\u4E48\u6837\u7684\u6548\u679C\uFF0C\u8FD8\u662F\u4E00\u4E2A\u5F00\u653E\u95EE\u9898\u3002CTNMT \u8FD9\u7BC7\u8BBA\u6587\uFF0C\u4ECE\u4E09\u4E2A\u65B9\u9762\u4ECB\u7ECD\u8FD9\u4E2A\u95EE\u9898\uFF1A</p>\\n<ol>\\n<li>\u9884\u8BAD\u7EC3\u6280\u672F\uFF0C\u6BD4\u5982 BERT\u6216\u8005 GPT \u5728\u673A\u5668\u7FFB\u8BD1\u4E2D\u7684\u5E94\u7528\u5B58\u5728\u4EC0\u4E48\u6311\u6218\uFF1F</li>\\n<li>\u9488\u5BF9\u8FD9\u4E9B\u8C03\u6574\uFF0C\u9700\u8981\u600E\u4E48\u6700\u5927\u7A0B\u5EA6\u5229\u7528\u9884\u8BAD\u7EC3\u77E5\u8BC6\uFF1F</li>\\n<li>\u9884\u8BAD\u7EC3\u548C\u673A\u5668\u7FFB\u8BD1\u7684\u878D\u5408\u8FD8\u6709\u4EC0\u4E48\u6F5C\u529B\uFF1F</li>\\n</ol>\\n","headers":[{"level":2,"title":"\u9884\u8BAD\u7EC3\u6280\u672F\u5728\u673A\u5668\u7FFB\u8BD1\u9886\u57DF\u5B58\u5728\u7684\u6311\u6218\u2014-\u707E\u96BE\u6027\u9057\u5FD8","slug":"\u9884\u8BAD\u7EC3\u6280\u672F\u5728\u673A\u5668\u7FFB\u8BD1\u9886\u57DF\u5B58\u5728\u7684\u6311\u6218\u2014-\u707E\u96BE\u6027\u9057\u5FD8","link":"#\u9884\u8BAD\u7EC3\u6280\u672F\u5728\u673A\u5668\u7FFB\u8BD1\u9886\u57DF\u5B58\u5728\u7684\u6311\u6218\u2014-\u707E\u96BE\u6027\u9057\u5FD8","children":[]},{"level":2,"title":"\u6E10\u8FDB\u5F0F\u5B66\u4E60\u7B56\u7565--\u7F13\u89E3\u707E\u96BE\u6027\u9057\u5FD8\u95EE\u9898","slug":"\u6E10\u8FDB\u5F0F\u5B66\u4E60\u7B56\u7565-\u7F13\u89E3\u707E\u96BE\u6027\u9057\u5FD8\u95EE\u9898","link":"#\u6E10\u8FDB\u5F0F\u5B66\u4E60\u7B56\u7565-\u7F13\u89E3\u707E\u96BE\u6027\u9057\u5FD8\u95EE\u9898","children":[{"level":3,"title":"\u68AF\u5EA6\u63A7\u5236","slug":"\u68AF\u5EA6\u63A7\u5236","link":"#\u68AF\u5EA6\u63A7\u5236","children":[]},{"level":3,"title":"\u57FA\u4E8E\u95E8\u63A7\u5236\u7684\u878D\u5408","slug":"\u57FA\u4E8E\u95E8\u63A7\u5236\u7684\u878D\u5408","link":"#\u57FA\u4E8E\u95E8\u63A7\u5236\u7684\u878D\u5408","children":[]},{"level":3,"title":"\u6E10\u8FDB\u84B8\u998F\u7B56\u7565","slug":"\u6E10\u8FDB\u84B8\u998F\u7B56\u7565","link":"#\u6E10\u8FDB\u84B8\u998F\u7B56\u7565","children":[]}]},{"level":2,"title":"\u5B9E\u9A8C\u6548\u679C\u548C\u672A\u6765\u65B9\u5411","slug":"\u5B9E\u9A8C\u6548\u679C\u548C\u672A\u6765\u65B9\u5411","link":"#\u5B9E\u9A8C\u6548\u679C\u548C\u672A\u6765\u65B9\u5411","children":[]}],"git":{"createdTime":1663040715000,"updatedTime":1664668908000,"contributors":[{"name":"Lei Li","email":"lileicc@gmail.com","commits":2}]},"readingTime":{"minutes":6.22,"words":1866},"filePathRelative":"mt/ctnmt/README.md","localizedDate":"December 31, 2020"}');export{t as data};
