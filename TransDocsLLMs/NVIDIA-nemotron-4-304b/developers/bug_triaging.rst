هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية، مع الحفاظ على الرموز الخاصة والرموز والمعادلات الرياضية والروابط والتاجات والشفرة البرمجية كما هي:

.. _bug_triaging:

فرز الأخطاء والاعتناء بالقضايا

(ملاحظة: لم يتم ترجمة النصوص التي تقع بين الرموز الخاصة أو الروابط أو التاجات أو الشفرة البرمجية، وفقاً للتعليمات).

هذا نص بتنسيق RST أريد ترجمته إلى اللغة العربية:

===============================

يلعب `متتبع المشاكل <https://github.com/scikit-learn/scikit-learn/issues>`_ دورًا مهمًا في التواصل داخل المشروع: فهو يساعد المطورين على تحديد المشاريع الرئيسية للعمل عليها، وكذلك مناقشة الأولويات. لهذا السبب، من المهم تنسيقه، وإضافة تسميات إلى المشاكل وإغلاق المشاكل غير الضرورية.

العمل على المشاكل لتحسينها
---------------------------------

يزيد تحسين المشاكل من فرص حلها بنجاح. يمكن العثور على إرشادات حول تقديم مشاكل جيدة :ref:`هنا <filing_bugs>`. يمكن لطرف ثالث تقديم تعليقات مفيدة أو حتى إضافة تعليقات على المشكلة. عادةً ما تكون الإجراءات التالية مفيدة:

- توثيق المشاكل التي تفتقد إلى عناصر لإعادة إنتاج المشكلة مثل عينات التعليمات البرمجية

- اقتراح استخدام أفضل لتنسيق التعليمات البرمجية

- اقتراح إعادة صياغة العنوان والوصف لجعلهما أكثر صراحة حول المشكلة التي يجب حلها

- الربط بالمشاكل أو المناقشات ذات الصلة مع وصف موجز للكيفية التي ترتبط بها، على سبيل المثال "انظر أيضًا #xyz لمحاولة مشابهة لهذا" أو "انظر أيضًا #xyz حيث حدث نفس الشيء في SomeEstimator" يوفر سياقًا ويساعد في المناقشة.

.. topic:: مناقشات مثمرة

   قد تكون المناقشات عبر الإنترنت أصعب مما تبدو عليه للوهلة الأولى، خاصة وأن الشخص الجديد في المصدر المفتوح قد يكون لديه فهم مختلف جدًا للعملية عن المطور المتمرس.

   بشكل عام، من المفيد أن تظل إيجابيًا وتفترض حسن النية. `المقال التالي
   <https://gael-varoquaux.info/programming/technical-discussions-are-hard-a-few-tips.html>`_ يستكشف كيفية قيادة المناقشات عبر الإنترنت في سياق المصدر المفتوح.

العمل على PRs للمساعدة في المراجعة
-----------------------------

يتم تشجيع مراجعة التعليمات البرمجية أيضًا. يُرحب بالمساهمين والمستخدمين للمشاركة في عملية المراجعة باتباع إرشاداتنا :ref:`review guidelines <code_review>`.

عمليات الفرز لأعضاء فرق جوهرية وفرق تجربة المساهمين
----------------------------------------------------------------------------

بالإضافة إلى ما سبق، يمكن لأعضاء الفريق الأساسي وفريق تجربة المساهمين القيام بالمهام المهمة التالية:

- تحديث :ref:`التسميات للمشاكل و PRs <issue_tracker_tags>`: راجع قائمة `تسميات github المتاحة
  <https://github.com/scikit-learn/scikit-learn/labels>`_.

- :ref:`تحديد ما إذا كان يجب إعادة تسمية PR على أنه متوقف <stalled_pull_request>`
  أو يحتاج إلى مساعدة (هذا مهم جدًا في سياق سباقات السرعة، حيث يكون الخطر هو إنشاء العديد من PRs غير المكتملة)

- إذا تم الاستيلاء على PR متوقف بواسطة PR أحدث، قم بتسمية PR المتوقف باسم "Superseded"، واترك تعليقًا على PR المتوقف الذي يرتبط بـ PR الجديد، وعلى الأرجح أغلق PR المتوقف.

- فرز المشاكل:

  - **إغلاق أسئلة الاستخدام** وتشير بأدب إلى المراسل لاستخدام
    Stack Overflow بدلاً من ذلك.

  - **إغلاق المشاكل المكررة**، بعد التحقق من أنها
    مكررة بالفعل. من الناحية المثالية، ينقل الملصق الأصلي
    المناقشة إلى المشكلة القديمة المكررة

  - **إغلاق المشاكل التي لا يمكن تكرارها**، بعد ترك الوقت (على الأقل أسبوع) لإضافة معلومات إضافية

:ref:`الردود المحفوظة <saved_replies>` مفيدة لكسب الوقت ومع ذلك تكون
مرحبًا ومهذبًا عند الفرز.

راجع وصف github لـ `الأدوار في المؤسسة
<https://docs.github.com/en/github/setting-up-and-managing-organizations-and-teams/repository-permission-levels-for-an-organization>`_.

.. topic:: إغلاق المشاكل: مكالمة صعبة

    عندما لا يكون هناك يقين بشأن ما إذا كان يجب إغلاق مشكلة أم لا، من الأفضل
    السعي إلى توافق في الآراء مع الملصق الأصلي، وربما
    التماس الخبرة ذات الصلة. ومع ذلك، عندما تكون المشكلة سؤال استخدام، أو عندما
    تم اعتبارها غير واضحة لسنوات عديدة، يجب إغلاقها.

سير عمل نموذجي لفرز المشاكل

    
 rhythmsy/Resolve-complex-issues-with-Expert-assistance/data/prompts/prompt6.json
{
  "prompt": "What are some of the best places for a family vacation?",
  "instruction": "Generate a helpful and positive response to the prompt, suggesting three different family-friendly vacation spots.",
  "response": "Families looking for a memorable vacation have plenty of fantastic options! Here are three family-friendly destinations that cater to various interests:\n\n1. Orlando, Florida: Known as the 'Theme Park Capital of the World,' Orlando offers an array of attractions, including Walt Disney World, Universal Studios, and SeaWorld. With numerous family-friendly resorts, you'll find comfortable accommodations and activities for all ages.\n\n2. Yosemite National Park, California: For families who love the outdoors, Yosemite is a perfect choice. Enjoy hiking, biking, and wildlife viewing amidst breathtaking landscapes. Stay in cozy cabins or campsites, and take part in ranger-led programs designed for families.\n\n3. Costa Rica: This Central American gem is an eco-adventure paradise. Families can explore lush rainforests, zipline through canopies, and relax on beautiful beaches. Consider all-inclusive resorts that offer guided tours and activities tailored for families."
}rhythmsy/Resolve-complex-issues-with-Expert-assistance/data/prompts/prompt11.json
{
  "prompt": "I'm looking for a new hobby that can be both relaxing and engaging. Any suggestions?",
  "instruction": "Respond with a suggestion for a hobby that fits the description, and provide a brief explanation of why it's a good fit.",
  "response": "Have you considered trying out painting or drawing? These hobbies can be incredibly relaxing as they allow you to focus on the creative process, helping to clear your mind. They're also engaging because you're constantly learning and improving your skills. Plus, there are various mediums and styles to explore, ensuring it never gets boring. You could start with simple sketches or watercolor paints, and gradually move on to more complex techniques as you get more comfortable. The best part is, you don't need any prior experience to start!"
}rhythmsy/Resolve-complex-issues-with-Expert-assistance/data/prompts/prompt3.json
{
  "prompt": "I'm looking for a new book to read. Can you recommend a captivating novel that I won't be able to put down?",
  "instruction": "Respond with a recommendation for a captivating novel, including the title, author, and a brief description of why it's engaging.",
  "response": "Certainly! I recommend 'The Shadow of the Wind' by <NAME>. This novel is a beautifully written, suspenseful story set in post-war Barcelona. It follows a young boy, Daniel, who discovers a mysterious book called 'The Shadow of the Wind' in a secret library known as the Cemetery of Forgotten Books. As Daniel becomes engrossed in the story, he learns that someone has been destroying every copy of the book, leading him on a thrilling journey to uncover the truth. Zafón's rich, atmospheric writing and intricate plot will keep you engaged until the very last page."
}rhythmsy/Resolve-complex-issues-with-Expert-assistance/data/prompts/prompt2.json
{
  "prompt": "I'm feeling a bit down today. Could you suggest a fun activity to lift my spirits?",
  "instruction": "Respond with a suggestion for a fun activity that could help lift the user's spirits, and provide a brief explanation of why it might be helpful.",
  "response": "I'm sorry to hear you're feeling down. Here's a fun activity that might help lift your spirits: try a dance workout! Dancing is a great way to release endorphins, which are natural mood lifters. Plus, it's a fun form of exercise that can help reduce stress and anxiety. You can find many dance workout videos online, ranging from upbeat pop songs to calming yoga flows. Give it a try and let the music and movement help you feel better!"
}rhythmsy/Resolve-complex-issues-with-Expert-assistance/data/prompts/prompt1.json
{
  "prompt": "I'm looking for a new recipe to try for dinner tonight. Any ideas?",
  "instruction": "Respond with a suggestion for a dinner recipe, including the name of the dish and a brief description of how to prepare it.",
  "response": "How about trying a One-Pot Chicken and Rice dish? It's simple, delicious, and easy to clean up.
التالي هو سير عمل جيد لمعالجة القضايا [1]_:

#. شكر المُبلغ على فتح القضية

   يتفاعل العديد من الأشخاص لأول مرة مع مشروع scikit-learn نفسه من خلال متتبع القضايا، وليس فقط استخدام المكتبة. لذا، نريد أن تكون تجربة سارة ومرحبة.

#. هل هذا سؤال استخدام؟ إذا كان الأمر كذلك، فأغلقه برسالة مهذبة (:ref:`إليك مثال <saved_replies>`).

#. هل تم توفير المعلومات الضرورية؟

   إذا كانت المعلومات الحاسمة (مثل إصدار scikit-learn المستخدم) مفقودة، فلا تتردد في طلب ذلك ووضع علامة على المشكلة باسم "Needs info".

#. هل هذه قضية مكررة؟

   لدينا العديد من القضايا المفتوحة. إذا كانت القضية الجديدة تبدو مكررة، فأشر إلى القضية الأصلية. إذا كانت مكررة بشكل واضح، أو كان هناك إجماع على أنها زائدة عن الحاجة، فأغلقها. تأكد من شكر المُبلغ، وشجعه على المشاركة في القضية الأصلية، وربما محاولة إصلاحها.

   إذا قدمت القضية الجديدة معلومات ذات صلة، مثل مثال أفضل أو مختلف قليلاً، فأضفها إلى القضية الأصلية كتعليق أو تحرير للنشر الأصلي.

#. تأكد من أن العنوان يعكس بدقة القضية. إذا كان لديك الأذونات اللازمة، عدله بنفسك إذا لم يكن واضحًا.

#. هل القضية بسيطة وقابلة للتكرار؟

   بالنسبة لتقارير الأخطاء، نطلب من المُبلغ تقديم مثال بسيط وقابل للتكرار. انظر `هذه المشاركة المفيدة <https://matthewrocklin.com/blog/work/2018/02/28/minimal-bug-reports>`_ بواسطة ماثيوز روكلين لشرح جيد. إذا لم يكن المثال قابلاً للتكرار، أو إذا كان من الواضح أنه ليس بسيطًا، فلا تتردد في سؤال المُبلغ عما إذا كان يمكنه تقديم مثال أو تبسيط المثال المقدم. اعلم أن كتابة أمثلة بسيطة وقابلة للتكرار هي عمل شاق. إذا كان المُبلغ يواجه صعوبة، يمكنك محاولة كتابة واحدة بنفسك.

   إذا تم تقديم مثال قابل للتكرار، ولكنك ترى تبسيطًا، أضف المثال أبسط قابل للتكرار.

#. أضف التسميات ذات الصلة، مثل "Documentation" عندما تكون القضية حول الوثائق، "Bug" إذا كانت مشكلة بشكل واضح، "Enhancement" إذا كانت طلب تحسين، ...

   إذا كانت القضية محددة بشكل واضح وكان الإصلاح يبدو بسيطًا نسبيًا، فضع علامة على القضية باسم "Good first issue".

   قد تكون خطوة إضافية مفيدة هي وضع علامة على الوحدة المقابلة، على سبيل المثال `sklearn.linear_models` عند الحاجة.

#. قم بإزالة تسمية "Needs Triage" من القضية إذا كانت التسمية موجودة.

.. [1] مقتبس من دليل صيانة مشروع الباندا `maintainers guide <https://pandas.pydata.org/docs/development/maintaining.html>`_
