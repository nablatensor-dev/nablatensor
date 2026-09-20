package com.nablatensor.codegen;

import java.io.IOException;
import java.io.Writer;
import java.util.List;
import java.util.Set;
import javax.annotation.processing.AbstractProcessor;
import javax.annotation.processing.RoundEnvironment;
import javax.annotation.processing.SupportedAnnotationTypes;
import javax.annotation.processing.SupportedSourceVersion;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.Element;
import javax.lang.model.element.ElementKind;
import javax.lang.model.element.Modifier;
import javax.lang.model.element.TypeElement;
import javax.lang.model.element.VariableElement;

@SupportedAnnotationTypes("com.nablatensor.codegen.Of")
@SupportedSourceVersion(SourceVersion.RELEASE_25)
public final class OfProcessor extends AbstractProcessor {
  @Override public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment round) {
    for (Element element : round.getElementsAnnotatedWith(Of.class)) {
      if (element.getKind() != ElementKind.CLASS) {
        processingEnv.getMessager().printError("@Of is only supported on classes", element);
        continue;
      }
      generate((TypeElement) element);
    }
    return true;
  }

  private void generate(TypeElement type) {
    String packageName = processingEnv.getElementUtils().getPackageOf(type).getQualifiedName().toString();
    String name = type.getSimpleName().toString();
    String draft = name + "Builder";
    List<VariableElement> fields = type.getEnclosedElements().stream()
        .filter(e -> e.getKind() == ElementKind.FIELD && !e.getModifiers().contains(Modifier.STATIC))
        .map(VariableElement.class::cast).toList();
    try {
      try (Writer out = processingEnv.getFiler()
          .createSourceFile(packageName + "." + draft, type).openWriter()) {
        out.write("package " + packageName + ";\n\n");
        out.write("/** Generated named construction API. */\n");
        out.write("public final class " + draft + " {\n");
        for (VariableElement field : fields) out.write("  private " + field.asType() + " " + field.getSimpleName() + ";\n");
        out.write("\n  " + draft + "() {}\n\n");
        for (VariableElement field : fields) {
          String fieldName = field.getSimpleName().toString();
          out.write("  public " + draft + " " + fieldName + "(" + field.asType() + " value) {\n");
          out.write("    this." + fieldName + " = value;\n    return this;\n  }\n\n");
        }
        out.write("  public " + name + " build() {\n");
        for (VariableElement field : fields) if (!field.asType().getKind().isPrimitive()) {
          String fieldName = field.getSimpleName().toString();
          out.write("    if (" + fieldName + " == null) throw new IllegalStateException(\"Missing required value: " + fieldName + "\");\n");
        }
        out.write("    return " + name + ".create(");
        for (int i = 0; i < fields.size(); i++) {
          if (i > 0) out.write(", ");
          out.write(fields.get(i).getSimpleName().toString());
        }
        out.write(");\n  }\n}\n");
      }
    } catch (IOException exception) {
      processingEnv.getMessager().printError("Could not generate " + draft + ": " + exception.getMessage(), type);
    }
  }
}
