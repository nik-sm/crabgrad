/// NOTE - careful about borrow muts, since both LHS and RHS could be same Value
/// Thus, need to finish dealing with LHS before dealing with RHS
#[macro_export]
macro_rules! impl_binary_op {
    ($self:ident, $rhs:ident, $trait:ident, $method:ident, $func:ident, $operator:tt, $body:tt) =>
    (

        // Method-call style
        impl Value {
            fn $func(&$self, $rhs: &Value) -> Self {
                $body
            }
        }

        // Operations between two Value
        impl $trait<Value> for Value
        {
            type Output = Self;
            #[inline]
            fn $method($self: Value, $rhs: Value) -> Self::Output {
                $self.$func(&$rhs)
            }
        }
        impl $trait<&Value> for Value
        {
            type Output = Value;
            #[inline]
            fn $method($self: Value, $rhs: &Value) -> Self::Output {
                $self.$func($rhs)
            }
        }
        impl<'a> $trait<&Value> for &'a Value
        {
            type Output = Value;
            #[inline]
            fn $method($self: &'a Value, $rhs: &Value) -> Self::Output {
                $self.$func($rhs)
            }
        }
        impl<'a> $trait<Value> for &'a Value
        {
            type Output = Value;
            #[inline]
            fn $method($self: &'a Value, $rhs: Value) -> Self::Output {
                $self.$func(&$rhs)
            }
        }

        // Value on RHS, f64
        // TODO - deduplicate, possibly by using Into<Value>
        // except without a negative trait or some strategy with marker traits,
        // there becomes an issue of conflicting impl
        impl $trait<Value> for f64 {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: Value) -> Self::Output {
                Value::from(self).$func(&rhs)
            }
        }
        impl $trait<&Value> for f64 {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: &Value) -> Self::Output {
                Value::from(self).$func(rhs)
            }
        }
        impl $trait<Value> for &f64 {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: Value) -> Self::Output {
                Value::from(self).$func(&rhs)
            }
        }
        impl $trait<&Value> for &f64 {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: &Value) -> Self::Output {
                Value::from(self).$func(rhs)
            }
        }

        // Value on RHS, i64
        impl $trait<Value> for i64 {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: Value) -> Self::Output {
                Value::from(self).$func(&rhs)
            }
        }
        impl $trait<&Value> for i64 {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: &Value) -> Self::Output {
                Value::from(self).$func(rhs)
            }
        }
        impl $trait<Value> for &i64 {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: Value) -> Self::Output {
                Value::from(self).$func(&rhs)
            }
        }
        impl $trait<&Value> for &i64 {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: &Value) -> Self::Output {
                Value::from(self).$func(rhs)
            }
        }

        // Value on LHS, f64
        impl $trait<f64> for Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: f64) -> Self::Output {
                self.$func(&Value::from(rhs))
            }
        }
        impl $trait<f64> for &Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: f64) -> Self::Output {
                self.$func(&Value::from(rhs))
            }
        }
        impl $trait<&f64> for Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: &f64) -> Self::Output {
                self.$func(&Value::from(rhs))
            }
        }
        impl $trait<&f64> for &Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: &f64) -> Self::Output {
                self.$func(&Value::from(rhs))
            }
        }

        // Value on LHS, i64
        impl $trait<i64> for Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: i64) -> Self::Output {
                self.$func(&Value::from(rhs))
            }
        }
        impl $trait<i64> for &Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: i64) -> Self::Output {
                self.$func(&Value::from(rhs))
            }
        }
        impl $trait<&i64> for Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: &i64) -> Self::Output {
                self.$func(&Value::from(rhs))
            }
        }
        impl $trait<&i64> for &Value {
            type Output = Value;
            #[inline]
            fn $method(self, rhs: &i64) -> Self::Output {
                self.$func(&Value::from(rhs))
            }
        }

    )
}
