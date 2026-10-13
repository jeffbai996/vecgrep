import React from "react";

type Props = { title: string };

export function Header({ title }: Props) {
  return <h1>{title}</h1>;
}

export const Footer = ({ title }: Props) => {
  const year = new Date().getFullYear();
  return <footer>{title} {year}</footer>;
};
