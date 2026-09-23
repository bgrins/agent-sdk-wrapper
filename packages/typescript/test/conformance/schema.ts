// A JSON Schema checker for the keywords docs/fixtures/conformance-v1.schema.json
// uses; any other keyword throws, so the schema cannot outgrow it unnoticed.
type Schema = { [keyword: string]: unknown };
const annotations = new Set([
  "$schema",
  "$id",
  "$defs",
  "title",
  "description",
]);

const typeOf = (value: unknown): string =>
  value === null
    ? "null"
    : Array.isArray(value)
      ? "array"
      : Number.isInteger(value)
        ? "integer"
        : typeof value;

/** Errors for `value` against `schema`, as `<path>: <problem>`; `root` resolves `$ref`. */
export function schemaErrors(
  value: unknown,
  schema: Schema,
  root: Schema = schema,
  path = "",
): string[] {
  const errors: string[] = [];
  const fail = (problem: string) => errors.push(`${path || "/"}: ${problem}`);
  const sub = (inner: unknown, at: string, child: unknown) =>
    schemaErrors(inner, child as Schema, root, `${path}/${at}`);
  const branches = (rule: unknown) =>
    (rule as Schema[]).map((child) => schemaErrors(value, child, root, path));
  // A value no branch accepts reports the errors of the closest branch.
  const closest = (results: string[][]) =>
    results.reduce((best, next) => (next.length < best.length ? next : best));
  const object =
    typeOf(value) === "object" ? (value as Record<string, unknown>) : undefined;
  for (const [keyword, rule] of Object.entries(schema)) {
    if (annotations.has(keyword)) continue;
    const count = rule as number;
    switch (keyword) {
      case "$ref": {
        const name = String(rule).replace(/^#\/\$defs\//, "");
        const target = (root.$defs as Record<string, Schema>)[name];
        if (!target) throw new Error(`unresolved $ref ${String(rule)}`);
        errors.push(...schemaErrors(value, target, root, path));
        break;
      }
      case "type": {
        const actual = typeOf(value);
        if (actual !== rule && !(rule === "number" && actual === "integer"))
          fail(`expected ${String(rule)}, got ${actual}`);
        break;
      }
      case "const":
        if (JSON.stringify(value) !== JSON.stringify(rule))
          fail(`expected ${JSON.stringify(rule)}`);
        break;
      case "enum":
        if (
          !(rule as unknown[]).some(
            (item) => JSON.stringify(item) === JSON.stringify(value),
          )
        )
          fail(
            `${JSON.stringify(value)} is not one of ${JSON.stringify(rule)}`,
          );
        break;
      case "pattern":
        if (typeof value === "string" && !new RegExp(String(rule)).test(value))
          fail(`${JSON.stringify(value)} does not match ${String(rule)}`);
        break;
      case "minLength":
        if (typeof value === "string" && value.length < count)
          fail(`shorter than ${count}`);
        break;
      case "minimum":
        if (typeof value === "number" && value < count) fail(`below ${count}`);
        break;
      case "maximum":
        if (typeof value === "number" && value > count) fail(`above ${count}`);
        break;
      case "exclusiveMinimum":
        if (typeof value === "number" && value <= count)
          fail(`not above ${count}`);
        break;
      case "minItems":
        if (Array.isArray(value) && value.length < count)
          fail(`fewer than ${count} items`);
        break;
      case "maxItems":
        if (Array.isArray(value) && value.length > count)
          fail(`more than ${count} items`);
        break;
      case "items":
        if (Array.isArray(value))
          value.forEach((item, index) => {
            errors.push(...sub(item, String(index), rule));
          });
        break;
      case "minProperties":
        if (object && Object.keys(object).length < count)
          fail(`fewer than ${count} properties`);
        break;
      case "required":
        for (const key of rule as string[])
          if (object && !(key in object)) fail(`missing ${key}`);
        break;
      case "properties":
        for (const [key, child] of Object.entries(rule as Schema))
          if (object && key in object)
            errors.push(...sub(object[key], key, child));
        break;
      case "additionalProperties": {
        const known = Object.keys((schema.properties as Schema) ?? {});
        for (const [key, inner] of Object.entries(object ?? {}))
          if (!known.includes(key))
            if (rule === false) fail(`unknown field ${key}`);
            else errors.push(...sub(inner, key, rule));
        break;
      }
      case "propertyNames":
        for (const key of Object.keys(object ?? {}))
          errors.push(...sub(key, key, rule));
        break;
      case "allOf":
        for (const child of rule as Schema[])
          errors.push(...schemaErrors(value, child, root, path));
        break;
      case "anyOf":
      case "oneOf": {
        const results = branches(rule);
        const matched = results.filter((result) => !result.length).length;
        if (matched === 0) errors.push(...closest(results));
        else if (keyword === "oneOf" && matched > 1)
          fail(`matches ${matched} oneOf branches, not 1`);
        break;
      }
      default:
        throw new Error(`unsupported schema keyword ${keyword}`);
    }
  }
  return errors;
}
