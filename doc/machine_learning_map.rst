:html_theme.sidebar_secondary.remove:

.. _ml_map:

اختيار المجرب المناسب
====================

في كثير من الأحيان، قد يكون الجزء الأصعب في حل مشكلة تعلم الآلة هو إيجاد المجرب المناسب للمهمة. تتفاوت المجربات في مدى ملاءمتها لأنواع مختلفة من البيانات والمشاكل.

تم تصميم مخطط التدفق أدناه لإعطاء المستخدمين دليلًا تقريبيًا حول كيفية التعامل مع المشاكل المتعلقة بالمجربات التي يمكن تجربتها على البيانات. انقر على أي مجرب في المخطط أدناه لعرض وثائقه. يجب قراءة رمز 😭 على أنه "إذا لم يحقق هذا المجرب النتيجة المرجوة، فاتبع السهم وجرّب المجرب التالي". استخدم عجلة التمرير للتكبير والتصغير، وانقر واسحب للتحرك حول المخطط. يمكنك أيضًا تنزيل المخطط: :download:`ml_map.svg <images/ml_map.svg>`.

.. raw:: html

  <style>
    #sk-ml-map {
      height: 80vh;
      margin: 1.5rem 0;
    }

    #sk-ml-map svg {
      height: 100%;
      width: 100%;
      border: 2px solid var(--pst-color-border);
      border-radius: 0.5rem;
    }

    html[data-theme="dark"] #sk-ml-map svg {
      filter: invert(90%) hue-rotate(180deg);
    }
  </style>

  <script src="_static/scripts/vendor/svg-pan-zoom.min.js"></script>
  <script>
    document.addEventListener("DOMContentLoaded", function () {
      const beforePan = function (oldPan, newPan) {
        const gutterWidth = 100, gutterHeight = 100;
        const sizes = this.getSizes();

        // Compute pan limits
        const leftLimit = -((sizes.viewBox.x + sizes.viewBox.width) * sizes.realZoom) + gutterWidth;
        const rightLimit = sizes.width - gutterWidth - (sizes.viewBox.x * sizes.realZoom);
        const topLimit = -((sizes.viewBox.y + sizes.viewBox.height) * sizes.realZoom) + gutterHeight;
        const bottomLimit = sizes.height - gutterHeight - (sizes.viewBox.y * sizes.realZoom);

        return {
          x: Math.max(leftLimit, Math.min(rightLimit, newPan.x)),
          y: Math.max(topLimit, Math.min(bottomLimit, newPan.y))
        };
      };

      // Limit the pan
      svgPanZoom("#sk-ml-map svg", {
        zoomEnabled: true,
        controlIconsEnabled: true,
        fit: 1,
        center: 1,
        beforePan: beforePan,
      });
    });
  </script>

  <div id="sk-ml-map">

.. raw:: html
  :file: images/ml_map.svg

.. raw:: html

  </div>

(ملحوظة: لم تتم ترجمة الأجزاء البرمجية وتعليمات HTML و CSS وروابط التنزيل والمخطط نفسه وفقًا للتعليمات الواردة في السؤال.)
