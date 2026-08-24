"use client";

import * as React from "react";
import { type ControllerProps, type FieldPath, type FieldValues, Controller, FormProvider, useFormContext } from "react-hook-form";

import { cn } from "@/lib/utils";

import { Label } from "@/components/ui/label";

const Form = FormProvider;

type FormFieldContextValue<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
> = {
  name: TName;
};

const FormFieldContext = React.createContext<FormFieldContextValue>({} as FormFieldContextValue);

function FormField<
  TFieldValues extends FieldValues = FieldValues,
  TName extends FieldPath<TFieldValues> = FieldPath<TFieldValues>,
>({ ...props }: ControllerProps<TFieldValues, TName>) {
  return (
    <FormFieldContext.Provider value={{ name: props.name }}>
      <Controller {...props} />
    </FormFieldContext.Provider>
  );
}

function useFormField() {
  const fieldContext = React.useContext(FormFieldContext);
  const itemContext = React.useContext(FormItemContext);
  const { getFieldState, formState } = useFormContext();

  const fieldState = getFieldState(fieldContext.name, formState);

  const id = itemContext.id;

  return {
    name: fieldContext.name,
    formItemId: `${id}-form-item`,
    formDescriptionId: `${id}-form-item-description`,
    formMessageId: `${id}-form-item-message`,
    ...fieldState,
  };
}

type FormItemContextValue = {
  id: string;
};

const FormItemContext = React.createContext<FormItemContextValue>({} as FormItemContextValue);

function FormItem({ className, ...props }: React.ComponentProps<"div">) {
  const id = React.useId();
  return (
    <FormItemContext.Provider value={{ id }}>
      <div data-slot="form-item" className={cn("grid gap-2", className)} {...props} />
    </FormItemContext.Provider>
  );
}

function FormLabel({ className, ...props }: React.ComponentProps<typeof Label>) {
  const { error, formItemId } = useFormField();
  return (
    <Label
      data-slot="form-label"
      data-error={Boolean(error)}
      htmlFor={formItemId}
      className={cn("data-[error=true]:text-destructive", className)}
      {...props}
    />
  );
}

function FormControl(props: React.ComponentProps<"div">) {
  const { error, formItemId, formDescriptionId, formMessageId } = useFormField();
  const describedBy = error ? `${formDescriptionId} ${formMessageId}` : formDescriptionId;
  const { children, ...rest } = props;

  if (React.isValidElement(children)) {
    return React.cloneElement(
      children as React.ReactElement<Record<string, unknown>>,
      {
        id: formItemId,
        "aria-describedby": describedBy,
        "aria-invalid": error ? true : undefined,
      },
    );
  }

  return (
    <div
      data-slot="form-control"
      id={formItemId}
      aria-describedby={describedBy}
      aria-invalid={Boolean(error)}
      {...rest}
    >
      {children}
    </div>
  );
}

function FormDescription({ className, ...props }: React.ComponentProps<"p">) {
  const { formDescriptionId } = useFormField();
  return (
    <p
      data-slot="form-description"
      id={formDescriptionId}
      className={cn("text-xs text-muted-foreground", className)}
      {...props}
    />
  );
}

function FormMessage({ className, ...props }: React.ComponentProps<"p">) {
  const { error, formMessageId } = useFormField();
  const body = error ? String(error.message ?? "") : props.children;
  if (!body) return null;
  return (
    <p
      data-slot="form-message"
      id={formMessageId}
      role="alert"
      className={cn("text-xs text-destructive", className)}
      {...props}
    >
      {body}
    </p>
  );
}

export { useFormField, Form, FormItem, FormLabel, FormControl, FormDescription, FormMessage, FormField };
